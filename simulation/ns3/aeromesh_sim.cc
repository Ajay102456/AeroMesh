// AeroMesh ns-3 packet-level simulation.
//
// Models the networking layer of the AeroMesh design described in the
// manuscript: a UAV fleet forming an 802.11 ad-hoc mesh, an endogenous
// per-vehicle failure process, and (in "aeromesh" mode) a self-healing
// mobility controller that repositions surviving vehicles toward gaps left
// by lost neighbors. Ground demand nodes generate three priority classes of
// UDP traffic toward a fixed gateway through the mesh. Real 802.11 MAC
// contention, real AODV/OLSR routing, and real propagation loss are used
// throughout; only the mobility control law and the failure schedule are
// AeroMesh-specific additions on top of stock ns-3.
//
// Modes:
//   aodv       - AODV routing, random-waypoint mobility, no self-healing
//   olsr       - OLSR routing, random-waypoint mobility, no self-healing
//   aeromesh   - AODV routing, self-healing repositioning mobility
//
// Usage:
//   ./aeromesh_sim --mode=aeromesh --nUav=32 --failLevel=1 --scenario=ph \
//                   --seed=1 --simTime=300 --out=results/run.csv

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/olsr-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <map>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("AeroMeshSim");

// ---------------------------------------------------------------------
// Self-healing beacon application (Gossip-Mesh, simplified).
//
// Each alive UAV broadcasts a 1 Hz UDP beacon carrying its own position.
// It tracks the most recent beacon time of every neighbor it has heard
// from. If a previously-heard neighbor goes silent for longer than the
// dead-time threshold, this node (acting as the nearest lightly loaded
// neighbor, approximated here by "any node that still hears at least one
// other live neighbor") steps its position toward the midpoint between
// its current position and the lost neighbor's last known position,
// capped by a maximum reposition speed. This is a direct, intentionally
// simplified implementation of the reposition branch of Algorithm 1
// (Gossip-Mesh) in the manuscript; the reroute and promote branches are
// not separately modeled because at the networking layer their effect is
// already captured by AODV's own route repair.
// ---------------------------------------------------------------------
class SelfHealingApp : public Application {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("SelfHealingApp")
      .SetParent<Application>()
      .AddConstructor<SelfHealingApp>();
    return tid;
  }

  void Setup(Ptr<Node> node, uint32_t nodeId, uint16_t port, double deadTime,
             double maxSpeed, Vector gatewayPos, bool healingEnabled,
             Time stopAfter) {
    m_node = node;
    m_nodeId = nodeId;
    m_port = port;
    m_deadTime = deadTime;
    m_maxSpeed = maxSpeed;
    m_gatewayPos = gatewayPos;
    m_healingEnabled = healingEnabled;
    m_deathTime = stopAfter;
    m_alive = true;
  }

private:
  virtual void StartApplication() {
    m_socket = Socket::CreateSocket(GetNode(), TypeId::LookupByName("ns3::UdpSocketFactory"));
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
    m_socket->Bind(local);
    m_socket->SetAllowBroadcast(true);
    m_socket->SetRecvCallback(MakeCallback(&SelfHealingApp::Receive, this));
    // Desynchronize beacon phase across vehicles so simultaneous
    // broadcasts from many nodes at the same instant do not collide at a
    // shared receiver (a known ns-3 YansWifiPhy edge case under dense,
    // perfectly synchronized broadcast traffic).
    Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable>();
    double offset = jitter->GetValue(0.0, 0.95);
    m_beaconEvent = Simulator::Schedule(Seconds(offset), &SelfHealingApp::SendBeacon, this);
  }

  virtual void StopApplication() {
    Simulator::Cancel(m_beaconEvent);
    if (m_socket) m_socket->Close();
  }

  void SendBeacon() {
    if (!m_alive) return;
    Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition();
    uint8_t buf[20];
    memcpy(buf, &m_nodeId, 4);
    float x = pos.x, y = pos.y, z = pos.z;
    double now = Simulator::Now().GetSeconds();
    memcpy(buf + 4, &x, 4);
    memcpy(buf + 8, &y, 4);
    memcpy(buf + 12, &z, 4);
    memcpy(buf + 16, &now, 4);
    Ptr<Packet> p = Create<Packet>(buf, 20);
    InetSocketAddress dst = InetSocketAddress(Ipv4Address("255.255.255.255"), m_port);
    m_socket->SendTo(p, 0, dst);

    CheckNeighborTimeouts();
    m_beaconEvent = Simulator::Schedule(Seconds(1.0), &SelfHealingApp::SendBeacon, this);
  }

  void Receive(Ptr<Socket> socket) {
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from))) {
      if (packet->GetSize() < 20) continue;
      uint8_t buf[20];
      packet->CopyData(buf, 20);
      uint32_t id; float x, y, z;
      memcpy(&id, buf, 4);
      memcpy(&x, buf + 4, 4);
      memcpy(&y, buf + 8, 4);
      memcpy(&z, buf + 12, 4);
      if (id == m_nodeId) continue;
      m_lastSeen[id] = Simulator::Now().GetSeconds();
      m_lastPos[id] = Vector(x, y, z);
    }
  }

  void CheckNeighborTimeouts() {
    if (!m_healingEnabled) return;
    double now = Simulator::Now().GetSeconds();
    bool haveLiveNeighbor = false;
    Vector gapTarget(0, 0, 0);
    int lostCount = 0;
    for (auto &kv : m_lastSeen) {
      double age = now - kv.second;
      if (age <= m_deadTime * 4) {
        haveLiveNeighbor = true;
      } else if (age <= m_deadTime * 4 + 2.0) {
        // Recently lost: treat as a repair target once.
        gapTarget.x += m_lastPos[kv.first].x;
        gapTarget.y += m_lastPos[kv.first].y;
        gapTarget.z += m_lastPos[kv.first].z;
        lostCount++;
      }
    }
    if (lostCount > 0 && haveLiveNeighbor) {
      gapTarget.x /= lostCount;
      gapTarget.y /= lostCount;
      gapTarget.z /= lostCount;
      Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
      Vector pos = mob->GetPosition();
      // Move partway toward the midpoint of the gap and the gateway,
      // capped by max reposition speed over this 1 s control tick.
      Vector target((gapTarget.x + m_gatewayPos.x) / 2.0,
                     (gapTarget.y + m_gatewayPos.y) / 2.0,
                     pos.z);
      Vector dir(target.x - pos.x, target.y - pos.y, 0);
      double dist = std::sqrt(dir.x * dir.x + dir.y * dir.y);
      if (dist > 1.0) {
        double step = std::min(m_maxSpeed * 1.0, dist);
        Vector newPos(pos.x + dir.x / dist * step, pos.y + dir.y / dist * step, pos.z);
        mob->SetPosition(newPos);
      }
    }
  }

public:
  void Kill() { m_alive = false; }

private:
  Ptr<Node> m_node;
  Ptr<Socket> m_socket;
  uint32_t m_nodeId{0};
  uint16_t m_port{9999};
  double m_deadTime{0.3};
  double m_maxSpeed{15.0};
  Vector m_gatewayPos;
  bool m_healingEnabled{false};
  bool m_alive{true};
  Time m_deathTime;
  EventId m_beaconEvent;
  std::map<uint32_t, double> m_lastSeen;
  std::map<uint32_t, Vector> m_lastPos;
};

// ---------------------------------------------------------------------

// Model vehicle loss by moving the dead UAV far outside radio range and
// silencing its applications, rather than calling WifiPhy::SetOffMode(),
// which can race with an in-flight reception on the same PHY and abort
// the simulation (a known ns-3 PHY state-machine limitation, not a
// property of the scenario being modeled). Moving the node to a
// sufficiently distant, fixed position is operationally equivalent for
// every metric collected here: the vehicle no longer transmits, relays,
// or is reachable by any other node.
static void KillNode(Ptr<Node> node, Ptr<SelfHealingApp> app) {
  Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
  if (mob) {
    mob->SetPosition(Vector(1.0e7, 1.0e7, 1.0e7));
  }
  if (app) app->Kill();
}

int main(int argc, char *argv[]) {
  uint32_t nUav = 32;
  uint32_t nDemand = 9;  // 3 per priority class
  std::string mode = "aeromesh";     // aodv | olsr | aeromesh
  uint32_t failLevel = 1;            // 0,1,2 increasing severity
  std::string scenario = "ph";       // ph | id | vn
  uint32_t seed = 1;
  double simTime = 300.0;
  std::string outFile = "results/run.csv";
  double areaSide = 3000.0;          // meters

  CommandLine cmd;
  cmd.AddValue("nUav", "Number of UAVs", nUav);
  cmd.AddValue("mode", "aodv|olsr|aeromesh", mode);
  cmd.AddValue("failLevel", "0=low,1=medium,2=high failure severity", failLevel);
  cmd.AddValue("scenario", "ph|id|vn", scenario);
  cmd.AddValue("seed", "RNG seed", seed);
  cmd.AddValue("simTime", "Simulation duration (s)", simTime);
  cmd.AddValue("out", "Output CSV path", outFile);
  cmd.AddValue("areaSide", "Square area side length (m)", areaSide);
  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed(seed);
  RngSeedManager::SetRun(seed);

  // Scenario geography tweaks: id (archipelagic) uses a larger, sparser
  // area; vn (gradual monsoon flood) uses a slower failure onset applied
  // via failLevel scaling below; ph (urban typhoon) is the base case.
  if (scenario == "id") areaSide *= 1.6;
  if (scenario == "vn") areaSide *= 1.2;

  bool healingEnabled = (mode == "aeromesh");
  std::string routing = (mode == "olsr") ? "olsr" : "aodv";

  NodeContainer uavNodes;
  uavNodes.Create(nUav);
  NodeContainer demandNodes;
  demandNodes.Create(nDemand);
  NodeContainer gatewayNode;
  gatewayNode.Create(1);

  NodeContainer allNodes;
  allNodes.Add(uavNodes);
  allNodes.Add(demandNodes);
  allNodes.Add(gatewayNode);

  // --- PHY / MAC: 802.11g ad hoc, real propagation loss + fading ---
  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211g);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue("ErpOfdmRate6Mbps"),
                                "ControlMode", StringValue("ErpOfdmRate6Mbps"),
                                "NonUnicastMode", StringValue("ErpOfdmRate6Mbps"));

  // Open-air UAV-to-UAV / UAV-to-ground link budget: licensed 500 mW
  // relay radio with modest antenna gain, open-sky path-loss exponent
  // (closer to free space than ground clutter), reference loss computed
  // for 2.4 GHz free-space at 1 m (Friis), plus Nakagami small-scale
  // fading for realism. This is a simplified stand-in for the
  // elevation-angle line-of-sight air-to-ground model cited in the
  // manuscript, which ns-3's stock propagation models do not implement
  // directly.
  YansWifiChannelHelper channel;
  channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                              "Exponent", DoubleValue(2.2),
                              "ReferenceLoss", DoubleValue(40.2));
  channel.AddPropagationLoss("ns3::NakagamiPropagationLossModel");
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());
  phy.Set("TxPowerStart", DoubleValue(27.0));
  phy.Set("TxPowerEnd", DoubleValue(27.0));
  phy.Set("TxGain", DoubleValue(5.0));
  phy.Set("RxGain", DoubleValue(5.0));
  phy.Set("RxSensitivity", DoubleValue(-93.0));

  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");
  NetDeviceContainer devices = wifi.Install(phy, mac, allNodes);

  // --- Mobility ---
  MobilityHelper mobUav;
  Ptr<UniformRandomVariable> rvX = CreateObject<UniformRandomVariable>();
  rvX->SetAttribute("Min", DoubleValue(0.0));
  rvX->SetAttribute("Max", DoubleValue(areaSide));

  mobUav.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
      "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=" + std::to_string(areaSide) + "]"),
      "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=" + std::to_string(areaSide) + "]"),
      "Z", StringValue("ns3::UniformRandomVariable[Min=80|Max=120]"));

  if (healingEnabled) {
    // Self-healing mode: position is held by ConstantPosition and moved
    // explicitly by SelfHealingApp's control law, rather than following a
    // pre-set trajectory.
    mobUav.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  } else {
    mobUav.SetMobilityModel("ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=5|Max=15]"),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=2.0]"),
        "PositionAllocator", StringValue("ns3::RandomBoxPositionAllocator"));
  }
  mobUav.Install(uavNodes);

  MobilityHelper mobFixed;
  mobFixed.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobFixed.Install(demandNodes);
  mobFixed.Install(gatewayNode);

  // Gateway at area center-edge; demand nodes spread at the far edge so
  // traffic must cross the mesh rather than reach the gateway directly.
  Ptr<MobilityModel> gwMob = gatewayNode.Get(0)->GetObject<MobilityModel>();
  Vector gwPos(areaSide / 2.0, 0.0, 10.0);
  gwMob->SetPosition(gwPos);
  for (uint32_t i = 0; i < nDemand; ++i) {
    Ptr<MobilityModel> m = demandNodes.Get(i)->GetObject<MobilityModel>();
    double frac = (double)i / std::max<uint32_t>(1, nDemand - 1);
    m->SetPosition(Vector(frac * areaSide, areaSide, 10.0));
  }

  // --- Routing + Internet stack ---
  InternetStackHelper internet;
  if (routing == "olsr") {
    OlsrHelper olsr;
    internet.SetRoutingHelper(olsr);
  } else {
    AodvHelper aodv;
    internet.SetRoutingHelper(aodv);
  }
  internet.Install(allNodes);

  Ipv4AddressHelper address;
  address.SetBase("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer interfaces = address.Assign(devices);

  // --- Self-healing beacon apps on UAVs ---
  std::vector<Ptr<SelfHealingApp>> healApps;
  for (uint32_t i = 0; i < nUav; ++i) {
    Ptr<SelfHealingApp> app = CreateObject<SelfHealingApp>();
    app->Setup(uavNodes.Get(i), i, 9999, 0.3, 20.0, gwPos, healingEnabled, Seconds(simTime));
    uavNodes.Get(i)->AddApplication(app);
    app->SetStartTime(Seconds(0.5));
    app->SetStopTime(Seconds(simTime));
    healApps.push_back(app);
  }

  // --- Node failure process ---
  // Mean lifetime (s) decreases with failure severity level, modeling the
  // combined environmental + battery hazard rate lambda_i described in the
  // manuscript's node-failure process. vn (gradual monsoon flood) stretches
  // the schedule to represent slower-onset infrastructure loss.
  double meanLifetime[3] = {240.0, 140.0, 80.0};
  double onsetScale = (scenario == "vn") ? 1.6 : 1.0;
  Ptr<ExponentialRandomVariable> lifeRv = CreateObject<ExponentialRandomVariable>();
  lifeRv->SetAttribute("Mean", DoubleValue(meanLifetime[failLevel] * onsetScale));
  for (uint32_t i = 0; i < nUav; ++i) {
    double deathT = lifeRv->GetValue();
    if (deathT < simTime - 5.0 && deathT > 10.0) {
      Simulator::Schedule(Seconds(deathT), &KillNode, uavNodes.Get(i), healApps[i]);
    }
  }

  // --- Priority traffic: 3 classes x 3 demand nodes each, toward gateway ---
  uint16_t basePort = 5000;
  ApplicationContainer sinkApps, clientApps;
  for (uint32_t cls = 0; cls < 3; ++cls) {
    uint16_t port = basePort + cls;
    PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    sinkApps.Add(sinkHelper.Install(gatewayNode.Get(0)));

    // class 0 = responder (highest rate/priority), 1 = medical, 2 = civilian
    double rateKbps = (cls == 0) ? 256.0 : (cls == 1) ? 128.0 : 64.0;
    uint8_t tos = (cls == 0) ? 0xB8 /* EF */ : (cls == 1) ? 0x60 /* AF */ : 0x00 /* BE */;

    for (uint32_t d = 0; d < 3; ++d) {
      uint32_t demandIdx = cls * 3 + d;
      if (demandIdx >= nDemand) break;
      OnOffHelper onoff("ns3::UdpSocketFactory",
          InetSocketAddress(interfaces.GetAddress(nUav + nDemand), port));
      onoff.SetAttribute("DataRate", DataRateValue(DataRate(std::to_string((uint32_t)rateKbps) + "kbps")));
      onoff.SetAttribute("PacketSize", UintegerValue(512));
      onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
      onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
      onoff.SetAttribute("Tos", UintegerValue(tos));
      ApplicationContainer app = onoff.Install(demandNodes.Get(demandIdx));
      app.Start(Seconds(2.0));
      app.Stop(Seconds(simTime - 1.0));
      clientApps.Add(app);
    }
  }
  sinkApps.Start(Seconds(0.0));
  sinkApps.Stop(Seconds(simTime));

  // --- Flow monitor for real packet-level measurement ---
  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

  Simulator::Stop(Seconds(simTime));
  Simulator::Run();

  monitor->CheckForLostPackets();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

  std::ofstream out(outFile);
  out << "mode,scenario,failLevel,nUav,seed,flowId,srcPort,dstPort,txPackets,rxPackets,lostPackets,"
      << "txBytes,rxBytes,delaySum,jitterSum,throughputKbps,meanDelayMs,pdr\n";

  for (auto &kv : stats) {
    Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(kv.first);
    const FlowMonitor::FlowStats &s = kv.second;
    double throughputKbps = 0.0, meanDelayMs = 0.0, pdr = 0.0;
    double duration = simTime - 2.0;
    if (duration > 0) throughputKbps = (s.rxBytes * 8.0 / 1000.0) / duration;
    if (s.rxPackets > 0) meanDelayMs = (s.delaySum.GetSeconds() / s.rxPackets) * 1000.0;
    if (s.txPackets > 0) pdr = (double)s.rxPackets / (double)s.txPackets;

    out << mode << "," << scenario << "," << failLevel << "," << nUav << "," << seed << ","
        << kv.first << "," << t.sourcePort << "," << t.destinationPort << ","
        << s.txPackets << "," << s.rxPackets << "," << s.lostPackets << ","
        << s.txBytes << "," << s.rxBytes << "," << s.delaySum.GetSeconds() << ","
        << s.jitterSum.GetSeconds() << "," << throughputKbps << "," << meanDelayMs << "," << pdr << "\n";
  }
  out.close();

  Simulator::Destroy();
  return 0;
}

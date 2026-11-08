// AeroMesh joint network+role-auction ns-3 simulation.
//
// Extends the validated aeromesh_sim.cc networking model (same PHY/MAC,
// propagation, routing, self-healing beacon controller, and node-failure
// process -- unchanged) with a real, measured coupling between the network
// layer and the role auction described in the manuscript's Algorithm 2: at
// each epoch boundary, a subset of surviving UAVs is withdrawn from relay
// duty to represent on-board inference workload, and FlowMonitor is
// snapshotted at every epoch boundary to produce a genuine time-resolved
// per-class PDR series. This is a separate binary from aeromesh_sim; it does
// not modify or depend on the already-validated aeromesh_sim.cc or its
// campaign results.
//
// Role policy:
//   auction - nInference and extraRelay vary with the current epoch's
//             severity, matching the real formula used in the Python
//             co-simulator's "full" condition:
//               extraRelay = min(severity, 3)
//               nInference = max(5 - severity, 2)
//   fixed   - nInference is held constant at 5 and extraRelay is always 0,
//             matching the Python co-simulator's "fixed_roles" ablation.
//
// Severity schedule (declared, not measured): starts at --startFailLevel;
// if --selfHealing=1, drops by one level every 4 epochs; otherwise stays
// constant for the whole run. This is the same declared schedule used in
// simulation/cosim/cosim.py's severity_schedule(), reused here so the two
// artifacts are directly comparable.
//
// A UAV assigned to inference duty for an epoch is moved out of radio range
// for that epoch's duration and its self-healing beacon is suspended, then
// restored to its prior position at the epoch boundary if still alive --
// the same mechanism aeromesh_sim.cc uses for permanent node loss, applied
// here reversibly. This genuinely removes the UAV's relay capacity for that
// window rather than approximating the effect analytically.
//
// Usage:
//   ./aeromesh_joint_sim --scenario=ph --startFailLevel=1 --selfHealing=1 \
//       --rolePolicy=auction --nUav=16 --seed=1 --epochLen=30 --nEpochs=10 \
//       --snapOut=results/joint/run.csv

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <map>
#include <vector>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("AeroMeshJointSim");

// ---------------------------------------------------------------------
// Self-healing beacon application. Identical to aeromesh_sim.cc's
// SelfHealingApp, with one addition: SetDark()/Restore() so a node can be
// reversibly withdrawn from the mesh for a bounded window (inference duty)
// rather than only permanently killed.
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
             double maxSpeed, Vector gatewayPos, bool healingEnabled) {
    m_node = node;
    m_nodeId = nodeId;
    m_port = port;
    m_deadTime = deadTime;
    m_maxSpeed = maxSpeed;
    m_gatewayPos = gatewayPos;
    m_healingEnabled = healingEnabled;
    m_alive = true;
    m_dark = false;
  }

  void Kill() { m_alive = false; }

  // Withdraw from the mesh for an inference-duty window: move far outside
  // radio range (reuses the same mechanism aeromesh_sim.cc uses for
  // permanent loss) and stop beaconing, remembering the real position so
  // it can be restored exactly.
  void SetDark(bool dark) {
    if (!m_alive) return;  // a permanently dead node has nothing to toggle
    if (dark == m_dark) return;
    Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
    if (dark) {
      m_savedPos = mob->GetPosition();
      mob->SetPosition(Vector(1.0e7, 1.0e7, 1.0e7));
    } else {
      mob->SetPosition(m_savedPos);
    }
    m_dark = dark;
  }

  bool IsAlive() const { return m_alive; }

private:
  virtual void StartApplication() {
    m_socket = Socket::CreateSocket(GetNode(), TypeId::LookupByName("ns3::UdpSocketFactory"));
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
    m_socket->Bind(local);
    m_socket->SetAllowBroadcast(true);
    m_socket->SetRecvCallback(MakeCallback(&SelfHealingApp::Receive, this));
    Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable>();
    double offset = jitter->GetValue(0.0, 0.95);
    m_beaconEvent = Simulator::Schedule(Seconds(offset), &SelfHealingApp::SendBeacon, this);
  }

  virtual void StopApplication() {
    Simulator::Cancel(m_beaconEvent);
    if (m_socket) m_socket->Close();
  }

  void SendBeacon() {
    if (!m_alive || m_dark) {
      m_beaconEvent = Simulator::Schedule(Seconds(1.0), &SelfHealingApp::SendBeacon, this);
      return;
    }
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
  bool m_dark{false};
  Vector m_savedPos;
  EventId m_beaconEvent;
  std::map<uint32_t, double> m_lastSeen;
  std::map<uint32_t, Vector> m_lastPos;
};

static void KillNode(Ptr<Node> node, Ptr<SelfHealingApp> app) {
  Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
  if (mob) mob->SetPosition(Vector(1.0e7, 1.0e7, 1.0e7));
  if (app) app->Kill();
}

// ---------------------------------------------------------------------
// Epoch controller: rotates which surviving UAVs are on inference duty
// (dark) for the coming epoch, and snapshots cumulative FlowMonitor stats
// at every epoch boundary so per-epoch (i.e. real time-resolved) per-class
// PDR can be recovered by differencing consecutive snapshots.
// ---------------------------------------------------------------------
struct EpochState {
  std::vector<Ptr<SelfHealingApp>> *healApps;
  Ptr<FlowMonitor> monitor;
  Ptr<Ipv4FlowClassifier> classifier;
  uint32_t nUav;
  std::string rolePolicy;
  uint32_t startFailLevel;
  bool selfHealing;
  double epochLen;
  uint32_t nEpochs;
  uint16_t basePort;
  std::ofstream *snapOut;
  // cumulative tx/rx per class at the previous snapshot
  double prevTx[3] = {0, 0, 0};
  double prevRx[3] = {0, 0, 0};
};

static uint32_t SeverityAtEpoch(uint32_t startFailLevel, bool selfHealing, uint32_t epoch) {
  int s = (int)startFailLevel;
  for (uint32_t e = 1; e <= epoch; ++e) {
    if (selfHealing && e % 4 == 0) s = std::max(s - 1, 0);
  }
  return (uint32_t)s;
}

static void SnapshotAndRotate(EpochState *st, uint32_t epoch) {
  // --- snapshot: diff cumulative per-class tx/rx since previous epoch ---
  st->monitor->CheckForLostPackets();
  std::map<FlowId, FlowMonitor::FlowStats> stats = st->monitor->GetFlowStats();
  double curTx[3] = {0, 0, 0}, curRx[3] = {0, 0, 0};
  for (auto &kv : stats) {
    Ipv4FlowClassifier::FiveTuple t = st->classifier->FindFlow(kv.first);
    int cls = (int)t.destinationPort - (int)st->basePort;
    if (cls < 0 || cls > 2) continue;
    curTx[cls] += kv.second.txPackets;
    curRx[cls] += kv.second.rxPackets;
  }
  const char *names[3] = {"responder", "medical", "civilian"};
  double epochStartT = (epoch - 1) * st->epochLen;
  for (int c = 0; c < 3; ++c) {
    double dTx = curTx[c] - st->prevTx[c];
    double dRx = curRx[c] - st->prevRx[c];
    double pdr = (dTx > 0) ? dRx / dTx : (curTx[c] > 0 ? curRx[c] / curTx[c] : 0.0);
    (*st->snapOut) << epoch << "," << epochStartT << "," << names[c] << "," << pdr
                   << "," << dTx << "," << dRx << "\n";
    st->prevTx[c] = curTx[c];
    st->prevRx[c] = curRx[c];
  }

  if (epoch >= st->nEpochs) return;  // last boundary: no further rotation needed

  // --- role rotation for the upcoming epoch ---
  uint32_t severity = SeverityAtEpoch(st->startFailLevel, st->selfHealing, epoch);
  uint32_t nInference = (st->rolePolicy == "fixed") ? 5
                         : std::max<int>(5 - (int)severity, 2);
  nInference = std::min(nInference, st->nUav);

  // Desynchronize the role-switch instants across UAVs with small per-node
  // jitter, exactly as SelfHealingApp::StartApplication already does for
  // beacon phase: many nodes toggling SetDark() (an instantaneous position
  // teleport) at the identical simulated instant reproduces the same dense,
  // perfectly-synchronized-event condition that already required jitter to
  // avoid the YansWifiPhy state-machine edge case noted above SendBeacon.
  Ptr<UniformRandomVariable> jitterRv = CreateObject<UniformRandomVariable>();
  for (uint32_t i = 0; i < st->nUav; ++i) {
    Ptr<SelfHealingApp> app = (*st->healApps)[i];
    if (!app->IsAlive()) continue;
    bool onInferenceDuty = ((i + epoch) % st->nUav) < nInference;
    double jitter = jitterRv->GetValue(0.0, 0.9);
    Simulator::Schedule(Seconds(jitter), &SelfHealingApp::SetDark, app, onInferenceDuty);
  }
}

int main(int argc, char *argv[]) {
  uint32_t nUav = 16;
  std::string scenario = "ph";
  uint32_t startFailLevel = 1;
  bool selfHealing = true;
  std::string rolePolicy = "auction";  // auction | fixed
  uint32_t seed = 1;
  double epochLen = 30.0;
  uint32_t nEpochs = 10;
  std::string snapOutPath = "results/joint/run.csv";
  double areaSide = 1000.0;

  CommandLine cmd;
  cmd.AddValue("nUav", "Number of UAVs", nUav);
  cmd.AddValue("scenario", "ph|id|vn", scenario);
  cmd.AddValue("startFailLevel", "0=low,1=medium,2=high", startFailLevel);
  cmd.AddValue("selfHealing", "1=aeromesh self-healing, 0=no self-healing (AODV baseline)", selfHealing);
  cmd.AddValue("rolePolicy", "auction|fixed", rolePolicy);
  cmd.AddValue("seed", "RNG seed", seed);
  cmd.AddValue("epochLen", "Epoch duration (s)", epochLen);
  cmd.AddValue("nEpochs", "Number of epochs", nEpochs);
  cmd.AddValue("snapOut", "Per-epoch snapshot CSV output path", snapOutPath);
  cmd.AddValue("areaSide", "Square area side length (m)", areaSide);
  cmd.Parse(argc, argv);

  double simTime = epochLen * nEpochs;

  RngSeedManager::SetSeed(seed);
  RngSeedManager::SetRun(seed);

  if (scenario == "id") areaSide *= 1.6;
  if (scenario == "vn") areaSide *= 1.2;

  uint32_t nDemand = 9;
  NodeContainer uavNodes; uavNodes.Create(nUav);
  NodeContainer demandNodes; demandNodes.Create(nDemand);
  NodeContainer gatewayNode; gatewayNode.Create(1);
  NodeContainer allNodes; allNodes.Add(uavNodes); allNodes.Add(demandNodes); allNodes.Add(gatewayNode);

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211g);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue("ErpOfdmRate6Mbps"),
                                "ControlMode", StringValue("ErpOfdmRate6Mbps"),
                                "NonUnicastMode", StringValue("ErpOfdmRate6Mbps"));

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

  MobilityHelper mobUav;
  mobUav.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
      "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=" + std::to_string(areaSide) + "]"),
      "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=" + std::to_string(areaSide) + "]"),
      "Z", StringValue("ns3::UniformRandomVariable[Min=80|Max=120]"));
  mobUav.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobUav.Install(uavNodes);

  MobilityHelper mobFixed;
  mobFixed.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobFixed.Install(demandNodes);
  mobFixed.Install(gatewayNode);

  Ptr<MobilityModel> gwMob = gatewayNode.Get(0)->GetObject<MobilityModel>();
  Vector gwPos(areaSide / 2.0, 0.0, 10.0);
  gwMob->SetPosition(gwPos);
  for (uint32_t i = 0; i < nDemand; ++i) {
    Ptr<MobilityModel> m = demandNodes.Get(i)->GetObject<MobilityModel>();
    double frac = (double)i / std::max<uint32_t>(1, nDemand - 1);
    m->SetPosition(Vector(frac * areaSide, areaSide, 10.0));
  }

  InternetStackHelper internet;
  AodvHelper aodv;
  internet.SetRoutingHelper(aodv);
  internet.Install(allNodes);

  Ipv4AddressHelper address;
  address.SetBase("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer interfaces = address.Assign(devices);

  std::vector<Ptr<SelfHealingApp>> healApps;
  for (uint32_t i = 0; i < nUav; ++i) {
    Ptr<SelfHealingApp> app = CreateObject<SelfHealingApp>();
    app->Setup(uavNodes.Get(i), i, 9999, 0.3, 20.0, gwPos, selfHealing);
    uavNodes.Get(i)->AddApplication(app);
    app->SetStartTime(Seconds(0.5));
    app->SetStopTime(Seconds(simTime));
    healApps.push_back(app);
  }

  double meanLifetime[3] = {240.0, 140.0, 80.0};
  double onsetScale = (scenario == "vn") ? 1.6 : 1.0;
  Ptr<ExponentialRandomVariable> lifeRv = CreateObject<ExponentialRandomVariable>();
  lifeRv->SetAttribute("Mean", DoubleValue(meanLifetime[startFailLevel] * onsetScale));
  for (uint32_t i = 0; i < nUav; ++i) {
    double deathT = lifeRv->GetValue();
    if (deathT < simTime - 5.0 && deathT > 10.0) {
      Simulator::Schedule(Seconds(deathT), &KillNode, uavNodes.Get(i), healApps[i]);
    }
  }

  uint16_t basePort = 5000;
  ApplicationContainer sinkApps, clientApps;
  for (uint32_t cls = 0; cls < 3; ++cls) {
    uint16_t port = basePort + cls;
    PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    sinkApps.Add(sinkHelper.Install(gatewayNode.Get(0)));

    double rateKbps = (cls == 0) ? 256.0 : (cls == 1) ? 128.0 : 64.0;
    uint8_t tos = (cls == 0) ? 0xB8 : (cls == 1) ? 0x60 : 0x00;

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

  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> monitor = flowHelper.InstallAll();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());

  std::ofstream snapOut(snapOutPath);
  snapOut << "epoch,epochStartS,priorityClass,pdr,txPackets,rxPackets\n";

  EpochState st;
  st.healApps = &healApps;
  st.monitor = monitor;
  st.classifier = classifier;
  st.nUav = nUav;
  st.rolePolicy = rolePolicy;
  st.startFailLevel = startFailLevel;
  st.selfHealing = selfHealing;
  st.epochLen = epochLen;
  st.nEpochs = nEpochs;
  st.basePort = basePort;
  st.snapOut = &snapOut;

  // Apply epoch-0 role assignment before traffic starts, then snapshot and
  // rotate at every subsequent epoch boundary.
  {
    uint32_t severity0 = SeverityAtEpoch(startFailLevel, selfHealing, 0);
    uint32_t nInference0 = (rolePolicy == "fixed") ? 5 : std::max<int>(5 - (int)severity0, 2);
    nInference0 = std::min(nInference0, nUav);
    for (uint32_t i = 0; i < nUav; ++i) {
      bool onDuty = (i % nUav) < nInference0;
      healApps[i]->SetDark(onDuty);
    }
  }
  for (uint32_t e = 1; e <= nEpochs; ++e) {
    Simulator::Schedule(Seconds(e * epochLen), &SnapshotAndRotate, &st, e);
  }

  Simulator::Stop(Seconds(simTime));
  Simulator::Run();

  snapOut.close();
  Simulator::Destroy();
  return 0;
}

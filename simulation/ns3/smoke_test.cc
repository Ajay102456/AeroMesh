#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/olsr-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

int main(int argc, char *argv[]) {
  NodeContainer nodes;
  nodes.Create(4);

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211g);
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());
  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue("OfdmRate6Mbps"));
  NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

  MobilityHelper mobility;
  mobility.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
      "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=500]"),
      "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=500]"),
      "Z", StringValue("ns3::UniformRandomVariable[Min=50|Max=120]"));
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.Install(nodes);

  AodvHelper aodv;
  InternetStackHelper internet;
  internet.SetRoutingHelper(aodv);
  internet.Install(nodes);

  Ipv4AddressHelper address;
  address.SetBase("10.0.0.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = address.Assign(devices);

  std::cout << "ns-3 smoke test OK, node0 addr = " << interfaces.GetAddress(0) << std::endl;
  Simulator::Destroy();
  return 0;
}

//
// MultiCell_X2Mesh_19gNodeB_NR.ned
// Full X2-meshed NR network with 19 gNodeBs for 5G standalone simulations
// Converted from LTE to NR standalone architecture
// Created for Nevers city-scale deployment
//

package simu5g.simulations.NR.networks;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.NR.gNodeB;
import simu5g.nodes.NR.NRUe;
import simu5g.nodes.Upf;
import simu5g.world.radio.LteChannelControl;


network MultiCell_X2Mesh_19gNodeB_NR
{
    parameters:
        int numUe = default(1);
        @display("i=block/network2;bgb=2000,2000;bgi=background/terrain,s");
        
    submodules:
        channelControl: LteChannelControl {
            @display("p=50,25;is=s");
        }
        routingRecorder: RoutingTableRecorder {
            @display("p=50,75;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=50,125");
        }
        binder: Binder {
            @display("p=50,175;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=50,225;is=s");
        }
        server: StandardHost {
            @display("p=1800,1000;is=n;i=device/server");
        }
        router: Router {
            @display("p=1600,1000;i=device/smallrouter");
        }
        upf: Upf {
            @display("p=1400,1000");
        }
        iUpf: Upf {
            @display("p=1200,1000");
        }
        
        // 19 gNodeBs (NR base stations)
        gNodeB[19]: gNodeB {
            @display("p=1000,1000;is=vl");
        }
        
        // User Equipment (configurable number)
        ue[numUe]: NRUe {
            @display("p=500,500");
        }
        
    connections allowunconnected:
        // Data Network connections
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        
        // 5G Core Network connections (using intermediate UPF)
        upf.pppg++ <--> Eth10G <--> iUpf.pppg++;
        
        // Connect all gNodeBs to intermediate UPF
        for i=0..18 {
            iUpf.pppg++ <--> Eth10G <--> gNodeB[i].ppp;
        }
        
        // Full X2 mesh between all gNodeBs (171 connections total)
        // gNodeB 0 to all others
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[1].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[2].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[3].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[4].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[5].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[6].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[7].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[0].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 1 to remaining
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[2].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[3].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[4].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[5].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[6].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[7].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[1].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 2 to remaining
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[3].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[4].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[5].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[6].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[7].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[2].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 3 to remaining
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[4].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[5].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[6].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[7].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[3].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 4 to remaining
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[5].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[6].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[7].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[4].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 5 to remaining
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[6].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[7].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[5].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 6 to remaining
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[7].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[6].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 7 to remaining
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[8].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[7].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 8 to remaining
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[9].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[8].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 9 to remaining
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[10].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[9].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 10 to remaining
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[11].x2++;
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[10].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 11 to remaining
        gNodeB[11].x2++ <--> Eth10G <--> gNodeB[12].x2++;
        gNodeB[11].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[11].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[11].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[11].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[11].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[11].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 12 to remaining
        gNodeB[12].x2++ <--> Eth10G <--> gNodeB[13].x2++;
        gNodeB[12].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[12].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[12].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[12].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[12].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 13 to remaining
        gNodeB[13].x2++ <--> Eth10G <--> gNodeB[14].x2++;
        gNodeB[13].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[13].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[13].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[13].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 14 to remaining
        gNodeB[14].x2++ <--> Eth10G <--> gNodeB[15].x2++;
        gNodeB[14].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[14].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[14].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 15 to remaining
        gNodeB[15].x2++ <--> Eth10G <--> gNodeB[16].x2++;
        gNodeB[15].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[15].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 16 to remaining
        gNodeB[16].x2++ <--> Eth10G <--> gNodeB[17].x2++;
        gNodeB[16].x2++ <--> Eth10G <--> gNodeB[18].x2++;
        
        // gNodeB 17 to remaining
        gNodeB[17].x2++ <--> Eth10G <--> gNodeB[18].x2++;
}

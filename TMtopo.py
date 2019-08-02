#!/usr/bin/env python2
""" A small example showing the usage of Docker containers.
"""
#tcp-timestamp , icmp, udp
import time
from MaxiNet.Frontend import maxinet
from MaxiNet.Frontend import maxinet
from MaxiNet.Frontend.container import Docker
from mininet.topo import Topo
from mininet.node import OVSSwitch
topo = Topo()

t1 = topo.addHost("t1", cls=Docker, ip="10.0.0.1", dimage="tasnimul/sdnserver:latest")
t2 = topo.addHost("t2", cls=Docker, ip="10.0.0.2", dimage="tasnimul/sdnserver:latest")
t3 = topo.addHost("t3", cls=Docker, ip="10.0.0.3", dimage="tasnimul/randompacketgenerator:latest")
t4 = topo.addHost("t4", cls=Docker, ip="10.0.0.4", dimage="tasnimul/randompacketgenerator:latest")
t5 = topo.addHost("t5", cls=Docker, ip="10.0.0.5", dimage="tasnimul/randompacketgenerator:latest")
t6 = topo.addHost("t6", cls=Docker, ip="10.0.0.6", dimage="tasnimul/randompacketgenerator:latest")
t7 = topo.addHost("t7", cls=Docker, ip="10.0.0.7", dimage="tasnimul/randompacketgenerator:latest")


#Creating Switches
s8 = topo.addSwitch("s8")

#Connecting hosts to switches
topo.addLink(t1, s8)
topo.addLink(t2, s8)
topo.addLink(t3, s8)
topo.addLink(t4, s8)
topo.addLink(t5, s8)
topo.addLink(t6, s8)
topo.addLink(t7, s8)



cluster = maxinet.Cluster()
exp = maxinet.Experiment(cluster, topo, switch=OVSSwitch)
exp.setup()

try:
	print exp.get_node("t1").cmd("ping -c 1 10.0.0.7")
	print exp.get_node("t2").cmd("ping -c 1 10.0.0.1")
	print exp.get_node("t3").cmd("ping -c 1 10.0.0.1")
	print exp.get_node("t4").cmd("ping -c 1 10.0.0.1")
	print exp.get_node("t5").cmd("ping -c 1 10.0.0.1")
	print exp.get_node("t6").cmd("ping -c 1 10.0.0.1")
	#print exp.get_node("t7").cmd("ping -c 1 10.0.0.1")
        time.sleep(5)
        exp.get_node("t1").cmd("service apache2 start && service apache2 enable")
	exp.get_node("t2").cmd("service apache2 start && service apache2 enable")
        #time.sleep(5)
        #exp.get_node("t4").cmd("python /root/randompacket.py icmp 10.0.0.3 &")
        #time.sleep(5)
	#exp.get_node("t1").cmd("echo 15000 15500 > /proc/sys/net/ipv4/ip_local_port_range")
	
	#exp.get_node("t1").cmd("perl /home/slowloris.pl/slowloris.pl -dns 10.0.0.11 -port 80 &")
	#time.sleep(5)
	#exp.get_node("t2").cmd("python /root/randompacket.py tcp-timestamp 10.0.0.3 &")
	

	exp.CLI(locals(), globals())

finally:
      exp.stop()

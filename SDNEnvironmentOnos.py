from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import operator
import requests
from requests.auth import HTTPBasicAuth
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_eager_execution
#Python Environment
class SDNEnvironment(py_environment.PyEnvironment):
  f=[]
#first define variables shape.Name of the variables are standard naming
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=10, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(12,), dtype=np.float32, minimum=[-1,-1,-1,-1,0,0,0,-1,-1,0,0,0],maximum=[1,1,1,1,1,1,1,1,1,1,1,1], name='observation')
    self._state = 0
    self.episode_ended = False
    self._postheader = {'Accept': 'application/json'}
    self._getheader = {'Content-Type': 'application/json', 'Accept': 'text/html'}
    deviceresponse = requests.get('http://127.0.0.1:8181/onos/v1/devices',
                                  auth=HTTPBasicAuth('onos', 'rocks')).json()
    self._applications= 6
    self._facp = 3000
    self._maxReward= 11

    self._currentFlow= 0
    self._MAC = '9A:7E:46:2A:B3:44' ### just randomly initializing with one of the Dst_host
    self._dest_currentflow = 0
    self._flowCount= 0
    self._deviceids = []
    for x in deviceresponse:
        for p in deviceresponse[x]:
            self._deviceids.append(str(p['id']))


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
      return self._observation_spec  #

  def _reset(self):
    self._reward_Flow = self.get_reward_State()
    self._currentFlow = self._reward_Flow[0]
    avg_host_flow = self._reward_Flow[2]
    active_hosts =  self._reward_Flow[3]
    TCP_Percentage = self._reward_Flow[4]
    UDP_Percentage = self._reward_Flow[5]
    ICMP_Percentage = self._reward_Flow[6]
    self._dest_currentflow= self._reward_Flow[7]
    dest_TCP = self._reward_Flow[8]
    dest_UDP = self._reward_Flow[9]
    dest_ICMP = self._reward_Flow[10]

    self._state = [(self._currentFlow / self._facp) if (self._currentFlow<=self._facp) else -1, 0.0, (avg_host_flow/self._facp) if ((active_hosts*avg_host_flow)<=self._facp) else -1,(active_hosts/self._facp) if (active_hosts<=self._facp) else -1, TCP_Percentage,
                       UDP_Percentage, ICMP_Percentage,(self._dest_currentflow / self._facp) if (self._dest_currentflow <=self._facp) else -1,0.0,(dest_TCP/ self._currentFlow) if (self._currentFlow > 0 ) else 0,
                        (dest_UDP/ self._currentFlow) if (self._currentFlow > 0) else 0,(dest_ICMP/ self._currentFlow) if (self._currentFlow > 0) else 0]
    return ts.restart(np.array(self._state, dtype=np.float32))

  def _step(self, action):
    self.postAction(action)
    time.sleep(10.0)
    #self.Flow()
    self._flowCount=self._flowCount +1

    # Make sure episodes don't go on forever.If 60 flows have reached end episode ..so after each 10 minutes approximately
    if self._flowCount == 40:
      self.episode_ended = True
      self._flowCount = 0
      self._reward_Flow = self.get_reward_State()
    else:
      pastFlow=self._currentFlow
      past_destHostFlow = self._dest_currentflow
      #self.postAction(action)
      self._reward_Flow= self.get_reward_State()
      self._currentFlow=self._reward_Flow[0]
      avg_host_flow = self._reward_Flow[2]
      active_hosts = self._reward_Flow[3]
      TCP_Percentage = self._reward_Flow[4]
      UDP_Percentage = self._reward_Flow[5]
      ICMP_Percentage = self._reward_Flow[6]
      self._dest_currentflow = self._reward_Flow[7]
      dest_TCP = self._reward_Flow[8]
      dest_UDP = self._reward_Flow[9]
      dest_ICMP = self._reward_Flow[10]
      deltaF = abs(self._currentFlow-pastFlow)
      delta_destHostFlow = abs(self._dest_currentflow - past_destHostFlow)
      self._state = [(self._currentFlow / self._facp) if (self._currentFlow <= self._facp) else -1, (deltaF / self._facp) if (deltaF <= self._facp) else -1,(avg_host_flow/self._facp) if ((active_hosts*avg_host_flow)<=self._facp) else -1,
                     (active_hosts / self._facp) if (active_hosts <= self._facp) else -1,TCP_Percentage, UDP_Percentage, ICMP_Percentage,(self._dest_currentflow / self._facp) if (self._dest_currentflow <=self._facp) else -1,(delta_destHostFlow / self._facp) if (delta_destHostFlow <= self._facp) else -1,(dest_TCP/ self._currentFlow) if (self._currentFlow > 0 ) else 0,
                   (dest_UDP/ self._currentFlow) if (self._currentFlow > 0) else 0,(dest_ICMP/ self._currentFlow) if (self._currentFlow > 0) else 0]
    if self._currentFlow <= self._facp:
        if self._reward_Flow[1] > self._applications:
            reward = (self._reward_Flow[1]/self._maxReward)
        else:
            reward = ((2*self._reward_Flow[1] - self._applications)/self._maxReward)
    else:
        reward = -1
    if self.episode_ended :
        pastFlow = self._currentFlow
        past_destHostFlow = self._dest_currentflow
        self._currentFlow = self._reward_Flow[0]
        avg_host_flow = self._reward_Flow[2]
        active_hosts = self._reward_Flow[3]
        TCP_Percentage = self._reward_Flow[4]
        UDP_Percentage = self._reward_Flow[5]
        ICMP_Percentage = self._reward_Flow[6]
        self._dest_currentflow = self._reward_Flow[7]
        dest_TCP = self._reward_Flow[8]
        dest_UDP = self._reward_Flow[9]
        dest_ICMP = self._reward_Flow[10]
        deltaF = abs(self._currentFlow - pastFlow)
        delta_destHostFlow = abs(self._dest_currentflow - past_destHostFlow)
        self._state = [(self._currentFlow / self._facp) if (self._currentFlow <= self._facp) else -1,
                       (deltaF / self._facp) if (deltaF <= self._facp) else -1,
                       (avg_host_flow / self._facp) if ((active_hosts * avg_host_flow) <= self._facp) else -1,
                       (active_hosts / self._facp) if (active_hosts <= self._facp) else -1, TCP_Percentage, UDP_Percentage,
                       ICMP_Percentage, (self._dest_currentflow / self._facp) if (self._dest_currentflow <= self._facp) else -1,
                       (delta_destHostFlow / self._facp) if (delta_destHostFlow <= self._facp) else -1,
                       (dest_TCP / self._currentFlow) if (self._currentFlow > 0) else 0,
                       (dest_UDP / self._currentFlow) if (self._currentFlow > 0) else 0,
                       (dest_ICMP / self._currentFlow) if (self._currentFlow > 0) else 0]

        return ts.transition(np.array(self._state, dtype=np.float32), reward)
    else:
      return ts.transition(
          np.array(self._state, dtype=np.float32), reward)

  def postAction(self,postaction):
      self._postaction=postaction+1
      parameters = "/home/sendate/onos/apps/fwd/src/main/java/org/onosproject/fwd/DeepMonitor/Action_DeepMonitor.txt"
      self.flow_delete()
      with open(parameters, 'w+') as filehandle:
            filehandle.write(str(self._MAC)+","+str(self._postaction))

  def flow_delete(self):
      postheader = {'Accept': 'application/json'}
      url = 'http://127.0.0.1:8181/onos/v1/flows/of:0000000000000008'
      response = requests.get(url, headers=postheader, auth=HTTPBasicAuth('karaf', 'karaf'))
      if response.json().has_key('flows'):
          currentFlow = len(response.json()['flows'])
      for x in response.json():
          for p in response.json()[x]:
              match_field_list = p['selector']['criteria']
              for match_field in match_field_list:
                  if (str(match_field.values()[1])) == "ETH_DST":
                      eth_dst = (match_field.values()[0])
                      if eth_dst ==self._MAC:
                          flowdelete = requests.delete('http://127.0.0.1:8181/onos/v1/flows/of%3A0000000000000008/' + str(p['id']),
                                   auth=('onos', 'rocks'), headers=postheader)



  def get_reward_State(self):
  ### harcoded for experimental purpose ..can be variable
          Src_Dst_SET = set()
          src = ""
          dst = ""
          Source_Destination_Pair = ""
          criteria = list()
          TCP_Count = 0
          UDP_Count = 0
          ICMP_Count = 0
          fcap = 3000
          eth_dst_dic=dict()
          eth_dst_state_dic = dict()
          url = 'http://127.0.0.1:8181/onos/v1/flows/of:0000000000000008'
          response = requests.get(url, headers=self._postheader, auth=HTTPBasicAuth('karaf', 'karaf'))
          if response.json().has_key('flows'):
              currentFlow = len(response.json()['flows'])
          for x in response.json():
              for p in response.json()[x]:
                 match_field_list = p['selector']['criteria']
                 criteria.append(len(match_field_list))
                 for match_field in match_field_list:
                   if (str(match_field.values()[1])) == "ETH_DST":
                         eth_dst = (match_field.values()[0])
                         # print('dst', eth_dst)
                         eth_dict_value = eth_dst_dic.get(eth_dst, None)
                         eth_dst_state_dic_value = eth_dst_state_dic.get(eth_dst, None)
                         if (eth_dict_value == None) and (eth_dst_state_dic_value == None):
                             eth_dst_dic[eth_dst] = 1
                             eth_dst_state_dic[eth_dst] = [1, 0, 0, 0]
                         else:
                             eth_dst_dic[eth_dst] += 1
                             eth_dst_state_dic[eth_dst][0] = eth_dst_state_dic[eth_dst][0] + 1
                             for proto_match_field in match_field_list:
                                 if ((str(proto_match_field.values()[1])) == "IP_PROTO"):
                                     # print 'here'
                                     proto = (proto_match_field.values()[0])
                                     if int(proto) == 1:
                                         eth_dst_state_dic[eth_dst][3] = eth_dst_state_dic[eth_dst][3] + 1
                                     elif int(proto) == 6:
                                         eth_dst_state_dic[eth_dst][1] = eth_dst_state_dic[eth_dst][1] + 1
                                     elif int(proto) == 17:
                                         eth_dst_state_dic[eth_dst][2] = eth_dst_state_dic[eth_dst][2] + 1



 #overall switch status
                   if (str(src) != str(dst)) and str(src) != "" and str(dst) != "":
                      Source_Destination_Pair = str(src) + str(dst)
                   if ((str(match_field.values()[1])) == "IP_PROTO"):
                    src = (match_field.values()[0])
                    if int(src) == 1:
                      ICMP_Count = ICMP_Count + 1
                    elif int(src) == 6:
                      TCP_Count = TCP_Count + 1
                    elif int(src) == 17:
                      UDP_Count = UDP_Count + 1
                 Src_Dst_SET.add(Source_Destination_Pair)  ### for current setup this is not necessary

          active_hosts=len(eth_dst_dic)
          if active_hosts > 0:
            avg_host_flow = (sum(eth_dst_dic.values()))/active_hosts
          else :
             avg_host_flow = 0
          if currentFlow > 0:
              TCP_Percentage= TCP_Count/ currentFlow
              UDP_Percentage= UDP_Count/ currentFlow
              ICMP_Percentage= ICMP_Count/ currentFlow
          else:
              TCP_Percentage = 0
              UDP_Percentage = 0
              ICMP_Percentage = 0

          reward = (sum(criteria) / len(criteria))
          switch_State=[currentFlow,reward,avg_host_flow,active_hosts,TCP_Percentage,UDP_Percentage,ICMP_Percentage]
          #print (reward_State)
          threshold = fcap / avg_host_flow
          thresolded_new_dict = {key: val for key, val in eth_dst_dic.items() if val >= threshold}
          sorted_flow_dict = sorted(thresolded_new_dict.items(), key=operator.itemgetter(1), reverse=True)
          #print(sorted_flow_dict)
          if len(sorted_flow_dict)> 1:
              x = list(sorted_flow_dict)[0]
              if x[0] in eth_dst_state_dic:
                  self._MAC = x[0]
                  #print('at least 1')
                  final_state = switch_State + eth_dst_state_dic.get(x[0])
          else:
               #print('not')
               self._MAC = "None"
               final_state = switch_State + [0,0,0,0]
          #print('sent', self._MAC, final_state)
          return final_state








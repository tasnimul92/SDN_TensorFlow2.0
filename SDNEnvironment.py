from __future__ import division
from __future__ import absolute_import

from __future__ import print_function


import numpy as np
import time
import requests
from requests.auth import HTTPBasicAuth
import json
from tf_agents.environments import py_environment

from tf_agents.specs import array_spec

from tf_agents.trajectories import time_step as ts
import random
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
        shape=(6,), dtype=np.float32, minimum=[-1,-1,-1,0,0,0],maximum=[1,1,1,1,1,1], name='observation')
    self._state = 0
    self._episode_ended = False
    self.postheader = {'Accept': 'application/json'}
    self.getheader = {'Content-Type': 'application/json', 'Accept': 'text/html'}
    deviceresponse = requests.get('http://127.0.0.1:8181/onos/v1/devices',
                                  auth=HTTPBasicAuth('onos', 'rocks')).json()
    self.currentFlow= 0
    self.flowCount= 0
    self.deviceids = []
    for x in deviceresponse:
        for p in deviceresponse[x]:
            self.deviceids.append(str(p['id']))


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
      return self._observation_spec  #

  def _reset(self):
    reward_Flow = self.get_reward_State()
    self.currentFlow = reward_Flow[0]
    Source_Destination_IP_Pair = reward_Flow[2]
    TCP_Percentage = reward_Flow[3]
    UDP_Percentage = reward_Flow[4]
    ICMP_Percentage = reward_Flow[5]

    self._state = [(self.currentFlow / 3000) if (self.currentFlow<=3000) else -1, 0.0, (Source_Destination_IP_Pair/3000) if (Source_Destination_IP_Pair<=3000) else -1, TCP_Percentage,
                       UDP_Percentage, ICMP_Percentage]


    return ts.restart(np.array(self._state, dtype=np.float32))

  def _step(self, action):
    self.postAction(action)
    time.sleep(10.0)
    #self.Flow()
    self.flowCount=self.flowCount +1
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.If 60 flows have reached end episode ..so after each 10 minutes approximately
    if self.flowCount == 60:
      self._episode_ended = True
      self.flowCount = 0
    else:
      pastFlow=self.currentFlow
      #self.postAction(action)
      reward_Flow= self.get_reward_State()
      self.currentFlow=reward_Flow[0]
      Source_Destination_IP_Pair = reward_Flow[2]
      TCP_Percentage = reward_Flow[3]
      UDP_Percentage = reward_Flow[4]
      ICMP_Percentage = reward_Flow[5]
      deltaF = abs(self.currentFlow-pastFlow)
      self._state = [(self.currentFlow / 3000) if (self.currentFlow <= 3000) else -1, (deltaF / 3000) if (deltaF <= 3000) else -1,
                     (Source_Destination_IP_Pair / 3000) if (Source_Destination_IP_Pair <= 3000) else -1,
                     TCP_Percentage,
                     UDP_Percentage, ICMP_Percentage]

      if self.currentFlow <= 3000:
        reward = reward_Flow[1]
      else:
        reward = 0.0
    if self._episode_ended :
      return ts.termination(np.array(self._state, dtype=np.float32), reward)
    else:
      return ts.transition(
          np.array(self._state, dtype=np.float32), reward)

  def postAction(self,postaction):
      self.postaction=postaction
      parameters = "newmatchfield.json"
      with open(parameters, 'r') as filehandle:
          for newline in filehandle:
              self.f.append(newline)
      MatchField = self.f[postaction]
      flowdelete = requests.delete('http://127.0.0.1:8181/onos/v1/flows/application/org.onosproject.fwd',
                                   auth=('onos', 'rocks'), headers=self.postheader)
      time.sleep(0.05)
      newresponse = requests.post(
          'http://127.0.0.1:8181/onos/v1/configuration/org.onosproject.fwd.ReactiveForwarding?preset=false',
          data=MatchField, auth=('onos', 'rocks'), headers=self.getheader)



  def get_reward_State(self):
          Src_Dst_SET = set()
          src = ""
          dst = ""
          Source_Destination_Pair = ""
          criteria = list()
          TCP_Count = 0
          UDP_Count = 0
          ICMP_Count = 0
          url = 'http://127.0.0.1:8181/onos/v1/flows/of:0000000000000008'
          response = requests.get(url, headers=self.postheader, auth=HTTPBasicAuth('karaf', 'karaf'))
          if response.json().has_key('flows'):
              currentFlow = len(response.json()['flows'])
          for x in response.json():
              for p in response.json()[x]:
                 match_field_list = p['selector']['criteria']
                 criteria.append(len(match_field_list))
                 for match_field in match_field_list:

                   if ((str(match_field.values()[1])) == "IPV4_SRC"):
                       src = (match_field.values()[0])
                   if (str(match_field.values()[1])) == "IPV4_DST":
                      dst = (match_field.values()[0])
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
                 Src_Dst_SET.add(Source_Destination_Pair)
          TCP_Percentage= TCP_Count/ currentFlow
          UDP_Percentage= UDP_Count/ currentFlow
          ICMP_Percentage= ICMP_Count/ currentFlow
          reward = (sum(criteria) / len(criteria))
          reward_State=[currentFlow,reward,len(Src_Dst_SET),TCP_Percentage,UDP_Percentage,ICMP_Percentage]
          return reward_State








"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""
from objects.cc_base import CongestionControl

# We provided a simple algorithms about block selection to help you being familiar with this competition.
# In this example, it will select the block according to block's created time first and radio of rest life time to deadline secondly.
from player.block_selection import Solution as BlockSelection
import math

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

STATE_DOWN = 'SD'
STATE_UP = 'SU'
STATE_STA = 'SS'


class MySolution(BlockSelection, CongestionControl):

    def __init__(self):
        BlockSelection.__init__(self)
        CongestionControl.__init__(self)

        self.send_rate = 3000
        self.state_vec_list = []
        self.state = STATE_UP
        self.min_up_step = 125
        self.up_step = self.min_up_step
        self.up_pa = 2
        self.max_up_step = 500
        self.sta_step = self.up_step
        self.sta_hold_num = 0
        self.retrans_dict = dict()

        #采样间隔
        self.start_tmp_interval=3
        self.tmp_interval = self.start_tmp_interval
        self.max_tmp_interval = 7
        #采样开始时间
        self.tmp_time=0
        #目前采样数量
        self.tmp_num=0
        self.min_send_rate = 10

        self.max_send_rate = 10000
        self.min_tmp_interval = 3
        self.sample_space = 50

        self.weight_drop_rate = 0
        self.gamma = 0.9
        self.weight_latency = 0

        self.start_drop_rate = 0
        self.start_latency = 0
        
        #上一次调控时间
        self.last_time=0
        #下降幅度
        self.down_pa = 2
        self.now_time = 0
        self.wait_receive = False
        self.rtt = 0
        self.real_bandwidth = 3000
        #self.need_max_send_rate = 10000
        self.down_state_block_size = 3
        self.up_state_block_size = 6
        

    def select_block(self, cur_time, block_queue):

        if len(block_queue) == 0:
            return None
        best_block_idx = -1
        best_block_score = None
        best_block_remain_time = None
        re_best_block_idx = -1
        re_best_block_score = None
        re_best_block_remain_time = None
        priority_score = [3,2,1]
        ddl_list = []
        remain_packet_list = []
        remain_time_list = []
        score_list = []
        re_block_idx_list = []
        urgency_list = []

        for idx, item in enumerate(block_queue):
            block_score = 0

            block_id = item.block_info['Block_id']
            if item.retrans == True:
                # 已经开始重传的块没有其他未发送的packet了
                if block_id not in self.retrans_dict:
                    print('no retrans')
                else:
                    remain_packet = self.retrans_dict[block_id]
            else:
                remain_packet = item.block_info['Split_nums'] - item.offset
                if block_id in self.retrans_dict:
                    remain_packet += self.retrans_dict[block_id]
            priority = item.block_info['Priority']

            cost_per = priority_score[priority]/remain_packet

            ddl = item.block_info['Deadline']
            ddl_list.append(ddl)
            remain_time = ddl - (cur_time-item.block_info['Create_time'])
            urgency_list.append(remain_time/remain_packet)
            if remain_time <= 0:
                continue
            esitimate_suc = self.send_rate*remain_time / remain_packet
            trust_ground_pa = 0.25
            remain_packet_list.append(remain_packet)
            remain_time_list.append(remain_time)

            
            #if esitimate_suc > (1-trust_ground_pa) and esitimate_suc <= (1+trust_ground_pa):
                #block_score = cost_per*(1-trust_ground_pa + math.fabs(1-esitimate_suc)) + ddl - remain_time
            #    block_score = cost_per*esitimate_suc + 1 / remain_time
            #elif esitimate_suc > (1+trust_ground_pa):
            #    block_score = cost_per*(1+trust_ground_pa) + 1 / remain_time
            #else:
            #    block_score = cost_per*esitimate_suc
            
            block_score = cost_per
            score_list.append(block_score)

            if best_block_score is None or block_score > best_block_score:
                best_block_idx = idx
                best_block_score = block_score
                best_block_remain_time = remain_time
            elif block_score == best_block_score and remain_time < best_block_remain_time:
                best_block_idx = idx
                best_block_score = block_score
                best_block_remain_time = remain_time

            if not self.reliable_packet(remain_time, remain_packet):
                continue

            re_block_idx_list.append(idx)

            #if re_best_block_score is None or block_score > re_best_block_score:
                #re_best_block_idx = idx
                #re_best_block_score = block_score
                #re_best_block_remain_time = remain_time
            #elif block_score == re_best_block_score and remain_time < re_best_block_remain_time:
                #re_best_block_idx = idx
                #re_best_block_score = block_score
                #re_best_block_remain_time = remain_time
        
        #self.need_max_send_rate = sum(remain_packet_list)/(sum(remain_time_list)/len(remain_time_list))
        # 对可信赖队列进行处理
        # 考虑top3性价比的块，如果较低性价比的块更紧急则考虑先传它会不会让更高优先级的块变得不可信赖，如果不会，则选低一点性价比的
        #re_remain_time_list = [remain_time_list[idx] for idx in re_block_idx_list]
        #re_remain_packet_list = [remain_packet_list[idx] for idx in re_block_idx_list]
        if len(re_block_idx_list) != 0:
            re_score_list = [score_list[idx] for idx in re_block_idx_list]
            max_score = max(re_score_list)
            sorted_score_idx_list = sorted(range(len(re_score_list)), key=lambda k: re_score_list[k])
            size = self.up_state_block_size if self.state == STATE_UP else self.down_state_block_size
            length = size if len(re_block_idx_list) >= size else len(re_block_idx_list)
            re_block_top_idx_list = [re_block_idx_list[idx] for idx in sorted_score_idx_list[-length:]] # top3块
            re_remain_time_list = [remain_time_list[idx] for idx in re_block_top_idx_list]
            re_remain_packet_list = [remain_packet_list[idx] for idx in re_block_top_idx_list]
            score_top_list = [score_list[idx] for idx in re_block_top_idx_list]
            used_time = 0
            start_idx = 0
            for i in range(len(re_block_top_idx_list)):
                remain_time = re_remain_time_list[i]-used_time
                remain_packet = re_remain_packet_list[i]
                score = score_top_list[i]
                if score < max_score*1:
                    continue
                if self.reliable_packet(remain_time, remain_packet):
                    used_time += remain_packet/self.send_rate
                else:
                    start_idx = i
                    used_time -= sum([re_remain_time_list[j] for j in range(i)])
            re_best_block_idx = re_block_top_idx_list[-1]

        if re_best_block_idx == -1:
            sorted_urgency_list = sorted(range(len(urgency_list)), key=lambda k: urgency_list[k])
            max_urgency = max(urgency_list)
            size = self.up_state_block_size if self.state == STATE_UP else self.down_state_block_size
            length = size if len(sorted_urgency_list) >= size else len(sorted_urgency_list)
            block_top_idx_list = [idx for idx in sorted_urgency_list[-length:]] # top3块
            score_top_list = [score_list[idx] for idx in block_top_idx_list]
            urgency_top_list = [urgency_list[idx] for idx in block_top_idx_list]
            best_score = 0
            for i in range(len(block_top_idx_list)):
                if urgency_top_list[i] < max_urgency*0.95:
                    continue
                if score_top_list[i] >= best_score:
                    re_best_block_idx = block_top_idx_list[i]
                    best_score = score_top_list[i]

            #for idx, item in enumerate(block_queue):
                #block_score = 0

                #block_id = item.block_info['Block_id']
                #if item.retrans == True:
                    # 已经开始重传的块没有其他未发送的packet了
                    #if block_id not in self.retrans_dict:
                        #print('no retrans')
                    #else:
                        #remain_packet = self.retrans_dict[block_id]
                #else:
                    #remain_packet = item.block_info['Split_nums'] - item.offset
                    #if block_id in self.retrans_dict:
                        #remain_packet += self.retrans_dict[block_id]
                #priority = item.block_info['Priority']

                #cost_per = priority_score[priority]/remain_packet

                #ddl = item.block_info['Deadline']
                #remain_time = ddl - (cur_time-item.block_info['Create_time'])
               # block_score = remain_time/remain_packet
                #if re_best_block_score is None or block_score > re_best_block_score:
                    #re_best_block_idx = idx
                    #re_best_block_score = block_score
                    #re_best_block_remain_time = remain_time
                #elif block_score == re_best_block_score and remain_time < re_best_block_remain_time:
                    #re_best_block_idx = idx
                    #re_best_block_score = block_score
                    #re_best_block_remain_time = remain_time
        if re_best_block_idx == -1:
            re_best_block_idx = best_block_idx

        

        return re_best_block_idx

    def reliable_packet(self, remain_time, remain_packet):

        if math.floor(self.weight_drop_rate * remain_packet) == 0:
            rtt_latency = self.weight_latency #if self.weight_latency < self.start_latency else self.start_latency
        else:
            remain_packet += math.floor(self.weight_drop_rate * remain_packet)
            rtt_latency = 2*self.weight_latency #if self.weight_latency < self.start_latency else 2*self.start_latency
        remain_time -= rtt_latency
        if remain_time <= 0:
            return False
        if self.state != STATE_DOWN and remain_packet/remain_time < self.send_rate*0.9:
            return True
        elif self.state == STATE_DOWN and remain_packet/remain_time < self.send_rate*0.5:
            return True
        return False



    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm, when sender need to send packet.
        """

        return {
            "send_rate": self.send_rate,
            "extra":{
                    "Real_send_rate": self.send_rate,
                     "Send_time": cur_time
                     
                     #'Drop_rate': self.weight_drop_rate
                    #  'First_up': first
                    }
            }


    def cc_trigger(self, cur_time, event_info):

        
        old_send_rate = self.send_rate
        event_type = event_info["event_type"]
        event_time = cur_time
        send_delay = event_info['packet_information_dict']['Send_delay']
        packet_send_rate = event_info['packet_information_dict']['Extra']['Real_send_rate']
        latency = event_info['packet_information_dict']['Latency']
        packet_send_time = event_info['packet_information_dict']['Extra']['Send_time']
        # first_up_packet = event_info['packet_information_dict']['Extra']['First_up']
        # print("here!", event_time,event_type,event_info["packet_information_dict"]["Block_info"]["Block_id"],event_info["packet_information_dict"]["Packet_id"],event_info["packet_information_dict"]["Offset"])

        #self.weight_drop_rate = self.gamma*self.weight_drop_rate + (1-self.gamma) if event_type == EVENT_TYPE_DROP else self.gamma*self.weight_drop_rate
        if len(self.state_vec_list) > self.sample_space*2:
            self.weight_drop_rate = 0
            for vec in self.state_vec_list[-self.sample_space*2:]:
                if vec[1] == EVENT_TYPE_DROP:
                    self.weight_drop_rate += 1/(self.sample_space*2)
        self.weight_latency = self.gamma*self.weight_latency + (1-self.gamma)*latency
        #self.weight_latency = sum([vec[2] for vec in self.state_vec_list[-self.tmp_interval:]])/self.tmp_interval
        

        if self.start_latency == 0:
            self.start_latency = latency
        vec = [packet_send_rate, event_type, latency, packet_send_time, self.state, cur_time, event_info, self.weight_drop_rate, self.weight_latency]
        

        if send_delay == 0:
            retrans_block_id = event_info['packet_information_dict']['Block_info']['Block_id']
            if retrans_block_id in self.retrans_dict:
                self.retrans_dict[retrans_block_id] -= 1
                if self.retrans_dict[retrans_block_id] < 0:
                    print("error")
            else:
                print("error")

        if event_type == EVENT_TYPE_DROP:
            # 记录重传信息
            retrans_block_id = event_info['packet_information_dict']['Block_info']['Block_id']
            if retrans_block_id in self.retrans_dict:
                self.retrans_dict[retrans_block_id] += 1
            else:
                self.retrans_dict[retrans_block_id] = 1

        #self.state_vec_list.append(vec)
 
        if len(self.state_vec_list) == 0:
            self.state_vec_list.append(vec)
        else:
            for idx in range(1,len(self.state_vec_list)+1):
                # 保持列表按照包发送时间排序
                tmp_vec = self.state_vec_list[-idx]
                if tmp_vec[3] <= packet_send_time:
                    if idx == 1:
                        self.state_vec_list.append(vec)
                    else:
                        self.state_vec_list.insert(-(idx-1), vec)

                    break

        # 取历史最低的延迟作为rtt
        if self.rtt > latency or self.rtt == 0:
            self.rtt = latency

        #定时清理list中的元素
        if len(self.state_vec_list)>self.sample_space*5 :
            del self.state_vec_list[0:self.sample_space*2-1]

        #当前采样包的数量
        self.tmp_num=0
        for vec in self.state_vec_list:
            if vec[3]>=self.tmp_time:
                self.tmp_num =self.tmp_num+1
        
        # print(event_time,self.tmp_num)

        if cur_time-self.now_time>=1:
            self.now_time=math.floor(cur_time)
            if latency > 0.5:
                #self.up_step = self.min_up_step
                #self.send_rate+=self.up_step
                #self.state = STATE_UP
                #self.sta_hold_num = 0
                #self.tmp_time=event_time
                self.change_interval()
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate,
            }
        if self.state != STATE_DOWN and self.is_con_drop():
            # 太久以前造成的丢包不再作为参考
            self.send_rate = max(sum([vec[0] for vec in self.state_vec_list[-self.tmp_interval:]])/self.tmp_interval/self.down_pa, self.min_send_rate)
            self.state=STATE_DOWN
            self.sta_hold_num = 0
            self.start_drop_rate = self.weight_drop_rate
            self.start_latency = self.weight_latency
            self.tmp_time=event_time
            self.change_interval()
            self.real_bandwidth = self.send_rate*0.9
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate,
            }
        elif self.state == STATE_DOWN and self.tmp_num >= 3 and self.is_con_drop():
            self.send_rate = max(sum([vec[0] for vec in self.state_vec_list[-self.tmp_interval:]])/self.tmp_interval/self.down_pa, self.min_send_rate)
            #self.tmp_interval = self.start_tmp_interval
            self.sta_hold_num = 0
            self.tmp_time=event_time
            self.change_interval()
            self.real_bandwidth = self.send_rate*0.9
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate,
            }
        elif self.is_latency_up() or self.is_latency_down():
            self.state = STATE_DOWN
            #self.send_rate -= min(self.up_step, 0.3*self.send_rate)
            self.send_rate = max(self.get_estimate_rate(), self.min_send_rate)
            #self.send_rate = max(1/(self.state_vec_list[-1][2]-self.state_vec_list[-2][2]+1/self.state_vec_list[-2][0]), self.min_send_rate)
            self.change_interval()
            #self.tmp_interval = self.start_tmp_interval
            self.tmp_time=event_time
            
            self.start_drop_rate = self.weight_drop_rate
            self.start_latency = self.weight_latency
            return {
                "cwnd" : self.cwnd,
                "send_rate" : self.send_rate,
            }
        #if len(self.state_vec_list) > 50:
            #instance_list = [vec for vec in self.state_vec_list if vec[5]+0.1 >= self.state_vec_list[-1][5]]
            #send_hz = len(instance_list)/0.1
                
            #if send_hz < self.send_rate * 0.5:
            #    self.send_rate = self.send_rate
            #    self.change_interval()

            #    self.start_drop_rate = self.weight_drop_rate
            #    self.start_latency = self.weight_latency
            #    self.tmp_time=event_time
            #    return {
            #        "cwnd" : self.cwnd,
            #        "send_rate" : self.send_rate,
            #    }

       
        
        if self.state == STATE_UP:
            #采样结束
            #观测延迟情况
            
            if self.tmp_num > self.tmp_interval:
                
                self.up_step = min(self.up_step*self.up_pa, self.max_up_step)
                self.send_rate += self.up_step
                
                self.tmp_time = event_time
                self.start_latency = self.weight_latency
                self.real_bandwidth = self.send_rate*0.9
            elif sum([vec[2] for vec in self.state_vec_list[-self.tmp_interval:]])/self.tmp_interval > self.rtt+0.01:
                self.send_rate = self.send_rate
                self.state = STATE_DOWN
                self.tmp_time = event_time


                
        elif self.state == STATE_DOWN:
            # print(cur_time,"down")

            #if self.is_latency_down():
                #self.send_rate = self.get_estimate_rate()
                #self.start_drop_rate = self.weight_drop_rate
                #self.start_latency = self.weight_latency
                #self.sta_hold_num = 0
            
            if sum([vec[2] for vec in self.state_vec_list[-self.tmp_interval:]])/self.tmp_interval <= self.rtt+0.01:
                self.up_step = self.min_up_step
                self.send_rate += self.up_step
                self.state = STATE_UP
                self.start_latency = self.weight_latency

                self.tmp_time=event_time
            elif self.tmp_num > self.tmp_interval:
                self.send_rate *= 0.9
                self.tmp_time=event_time
                self.start_latency = self.weight_latency
            
        
        if self.send_rate<self.min_send_rate:
            self.send_rate=self.min_send_rate

        # set cwnd or sending rate in sender
        self.change_interval()
        return {
            "cwnd" : self.cwnd,
            "send_rate" : self.send_rate,
        }

 
    def get_estimate_rate(self):
        # 当判断延迟稳定上升或下降时才能使用这中方式估计带宽
        if len(self.state_vec_list) < self.tmp_interval:
            return self.send_rate
        latency_list = [vec[2] for vec in self.state_vec_list[-self.tmp_interval:]]
        create_time_list = [vec[3] for vec in self.state_vec_list[-self.tmp_interval:]]
        send_rate_list = [vec[0] for vec in self.state_vec_list[-self.tmp_interval:]]

        real_delta_latency = [latency_list[i]-(latency_list[i-1]-(create_time_list[i]-create_time_list[i-1])) for i in range(1,len(latency_list))]
        mean_delta = sum(real_delta_latency)/len(real_delta_latency)
        #mean_latency = sum(latency_list)/len(latency_list)
        assert(mean_delta > 0)

        extra_pa = 0.0098*(latency_list[-1]-self.rtt)

        real_rate = 1/(mean_delta+extra_pa)
        self.real_bandwidth = 1/real_delta_latency[0]
        return real_rate

    def is_con_drop(self):
        answer = False
        drop_event_list = [vec[1] for vec in self.state_vec_list[-3:]]
        for drop in drop_event_list:
            answer = True
            if drop != EVENT_TYPE_DROP:
                answer = False
                break
        return answer

    def is_drop_rate_up(self, ratio = 0.9):
        instance_list = [vec[7] for vec in self.state_vec_list[-self.tmp_interval:] if vec[7]>max(1.5*self.start_drop_rate, self.start_drop_rate+2/(self.max_tmp_interval*5))]
        if len(instance_list)/self.tmp_interval > ratio:
            return True
        return False

    def is_drop_rate_down(self, ratio = 0.9):
        instance_list = [vec[7] for vec in self.state_vec_list[-self.tmp_interval:] if vec[7]<min(self.start_drop_rate/1.5, self.start_drop_rate-2/(self.max_tmp_interval*5))]
        if len(instance_list)/self.tmp_interval > ratio:
            return True
        return False

    #def is_latency_up(self, ratio=0.9):
        # 是否有上涨趋势
        #latency_list = [vec[2] for vec in self.state_vec_list[-self.tmp_interval:]]
        #if self.check_up(latency_list, ratio):
        #    return True
        
        #latency_list=[vec[2] for vec in self.state_vec_list[-self.tmp_interval*3:]]
        #if self.check_up(latency_list, ratio):
        #    return True

        #latency_list = [vec[8] for vec in self.state_vec_list[-self.tmp_interval:] if vec[8] > self.start_latency]
        #if len(latency_list)/self.tmp_interval > ratio:
        #    return True
        #return False
    def is_latency_up(self):
        if len(self.state_vec_list) < self.tmp_interval:
            return False
        event_type_list = [1 for vec in self.state_vec_list[-self.tmp_interval:] if vec[1]==EVENT_TYPE_DROP]
        if len(event_type_list) != 0:
            return False
        latency_list = [vec[2] for vec in self.state_vec_list[-self.tmp_interval:]]
        create_time_list = [vec[3] for vec in self.state_vec_list[-self.tmp_interval:]]
        send_rate_list = [vec[0] for vec in self.state_vec_list[-self.tmp_interval:]]

        delta_latency = [latency_list[i]-latency_list[i-1] for i in range(1,len(latency_list))]
        for item in delta_latency:
            if item <= 1e-5:
                return False

        real_delta_latency = [latency_list[i]-(latency_list[i-1]-(create_time_list[i]-create_time_list[i-1])) for i in range(1,len(latency_list))]
        mean_delta = sum(real_delta_latency)/len(real_delta_latency)
        abs_delta_latency = [abs(latency-real_delta_latency[0]) for latency in real_delta_latency]
        #assert(mean_delta>=0)
        #if mean_delta <= 1e-5:
            #return False
        if real_delta_latency[0] <= 1e-5:
            return False
        for item in abs_delta_latency:
            if item > 1e-5:
                return False
        return True

    def check_up(self, list, ratio):
        up_case = 0
        down_case = 0
        length= len(list)
        step = math.ceil(len(list)/2)

        for i in range(math.floor(len(list)/2)):
            if list[i+step] > list[i]:
                up_case += 1
            else:
                down_case += 1
        if up_case / (up_case+down_case) >= ratio:
            return True
        return False

    def is_latency_down(self):
        # 是否有下降趋势
        if len(self.state_vec_list) < self.tmp_interval:
            return False
        latency_list = [vec[2] for vec in self.state_vec_list[-self.tmp_interval:]]
        create_time_list = [vec[3] for vec in self.state_vec_list[-self.tmp_interval:]]
        send_rate_list = [vec[0] for vec in self.state_vec_list[-self.tmp_interval:]]

        delta_latency = [latency_list[i]-latency_list[i-1] for i in range(1,len(latency_list))]
        for item in delta_latency:
            if item >= -1e-5:
                return False

        # 实际延迟梯度
        real_delta_latency = [latency_list[i] - (latency_list[i-1]-(create_time_list[i]-create_time_list[i-1])) for i in range(1,len(latency_list))]
        mean_delta = sum(real_delta_latency)/len(real_delta_latency)
        abs_delta_latency = [abs(latency-real_delta_latency[0]) for latency in real_delta_latency]
        #assert(mean_delta>=0)
        #if mean_delta <= 1e-5:
            #return False
        if real_delta_latency[0] <= 1e-5:
            return False
        for item in abs_delta_latency:
            if item > 1e-5:
                return False
        return True

    def check_down(self, list, ratio):
        up_case = 0
        down_case = 0
        length= len(list)
        step = math.ceil(len(list)/2)

        for i in range(math.floor(len(list)/2)):
            if list[i+step] > list[i]:
                up_case += 1
            else:
                down_case += 1
        if down_case / (up_case+down_case) >= ratio:
            return True
        return False

    def change_interval(self):
        if self.send_rate > self.max_send_rate:
            self.max_send_rate = self.send_rate
        k = float(self.max_tmp_interval-self.min_tmp_interval)/float(self.max_send_rate-self.min_send_rate)
        self.tmp_interval = self.min_tmp_interval + math.floor(k*(self.send_rate-self.min_send_rate))
    



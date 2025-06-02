import csv
import logging
import multiprocessing as mp
import os
import time
import subprocess
from typing import List
from scipy.stats import norm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from collections import deque
from simulator.master_QoE_4G.abr_trace import AbrTrace
from simulator.master_QoE_4G.base_abr import BaseAbr
from simulator.master_QoE_4G.schedulers import TestScheduler
from simulator.master_QoE_4G.utils import load_traces,load_trace, QoE_1,QoE_2,QoE_3,QoE_4,QoE_5

from simulator.master_QoE_4G.env import Environment
from simulator.master_QoE_4G.env_ZB import Environment as Env_zb
from simulator.master_QoE_4G import master_test_env_
from simulator.master_QoE_4G.ALONE import A_NN

from simulator.master_QoE_4G.constants import (
    A_DIM,
    BUFFER_NORM_FACTOR,
    CRITIC_LR_RATE,
    ACTOR_LR_RATE,
    DEFAULT_QUALITY,
    S_INFO,
    S_LEN,
    VIDEO_BIT_RATE,
    M_IN_K,
    VIDEO_CHUNK_LEN,
    MILLISECONDS_IN_SECOND,
    TOTAL_VIDEO_CHUNK,
    TRAIN_SEQ_LEN,
    SAMPLE_LEN,
    GAE_gamma,
    NUM_VARIANTS,
    QUEUE_LEN,
    REBUF_PENALTY,
    REBUF_penalty,
    SMOOTH_PENALTY,
)



from replay_memory import ReplayMemory

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU:",physical_devices)
for device in physical_devices:

    tf.config.experimental.set_memory_growth(device, True)





GAMMA = 0.99
RAND_RANGE = 1000
CHUNK_TIL_VIDEO_END_CAP = 48



#

def entropy_weight_decay_func(epoch):
    # linear decay
    # return np.maximum(-0.05/(10**4) * epoch + 0.5, 0.1)
    if epoch < 25000:
        return 3
    elif epoch < 50000:
        return 1
    else:
        return 0.3





class Pensieve(BaseAbr):
    abr_name = "pensieve"

    def __init__(self, model_path: str = "", s_info: int = 6, s_len: int = 8,
                 a_dim: int = 8, plot_flag: bool = False, train_mode=False):
        """Penseive
        Input state matrix shape: [s_info, s_len]

        Args
            model_path: pretrained model_path.
            s_info: number of features in input state matrix.
            s_len: number of past chunks.
            a_dim: number of actions in action space.
        """
        # tf.reset_default_graph()

        self.s_info = s_info
        self.s_len = s_len
        self.a_dim = a_dim
        self.model_path = model_path
        if self.s_info == 6 and self.s_len == 8 and self.a_dim == 8:
            print('use original pensieve')

        else:
            raise NotImplementedError
        self.plot_flag = plot_flag

        self.train_mode = train_mode
        if not self.train_mode:
            self.sess = tf.compat.v1.Session()
            self.actor = A_NN.ActorNetwork(
                self.sess,
                state_dim=[self.s_info, self.s_len],
                action_dim=self.a_dim,
                bitrate_dim=A_DIM
            )
            self.sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(max_to_keep=None)
            if self.model_path:
                saver.restore(self.sess, self.model_path)
                print("model restore")





    def _get_next_bitrate(self, state, last_bit_rate, actor):
        action_prob = actor.predict(state)
        action_cumsum = np.cumsum(action_prob)

        bit_rate = (
                action_cumsum
                > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
        ).argmax()
        return bit_rate, action_prob

    def master_trace_test(self, model_path:str, test_traces: str,dest_dir:str,
              video_size_file_dir: str, qoe_number: int):

        tf.compat.v1.reset_default_graph()

        rebuff_p = REBUF_penalty

        all_cooked_time, all_cooked_bw, all_file_names = load_trace(test_traces)




        test_env = master_test_env_.Environment(all_cooked_time=all_cooked_time,
                                   all_cooked_bw=all_cooked_bw, all_file_names=all_file_names,
                                   video_size_file=video_size_file_dir)

        test_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), VIDEO_BIT_RATE, 1, rebuff_p, SMOOTH_PENALTY, 0)

        if not os.path.exists(dest_dir ):
            os.makedirs(dest_dir )

        log_path = dest_dir + '/'  + all_file_names[test_env.trace_idx]
        log_file = open(log_path, 'wb')

        _, _, _, total_chunk_num, \
            bitrate_versions, rebuffer_penalty, smooth_penalty = test_env.get_env_info()

        # if not os.path.exists(SUMMARY_DIR):
        #     os.makedirs(SUMMARY_DIR)

        with tf.compat.v1.Session() as sess:
            entropy_weight = np.log(A_DIM)
            actor = A_NN.ActorNetwork(sess,
                                                  state_dim=[S_INFO, S_LEN],
                                                  action_dim=A_DIM,
                                                  learning_rate=ACTOR_LR_RATE,
                                                  bitrate_dim=A_DIM,
                                                  entropy_weight=entropy_weight,
                                                  name='actor'
                                                  )


            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()  # save neural net parameters

            nn_model = model_path

            # restore neural net parameters
            if nn_model is not None:  # nn_model is the path to file
                saver.restore(sess, nn_model)
                print("Model restored.")

            num_params = 0
            # for variable in tf.compat.v1.trainable_variables():
            #     shape = variable.get_shape()
            #     num_params += reduce(mul, [dim.value for dim in shape], 1)
            # print("num", num_params)

            time_stamp = 0

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch = [np.zeros((S_INFO, S_LEN))]
            a_batch = [action_vec]
            r_batch = []
            entropy_record = []

            video_count = 0

            while True:  # serve video forever
                # the action is from the last decision
                # this is to make the framework similar to the real
                delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain, \
         _ = test_env.get_video_chunk(bit_rate)

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                # reward is video quality - rebuffer penalty


                reward = bitrate_versions[bit_rate] / M_IN_K \
                         - rebuffer_penalty * rebuf \
                         - smooth_penalty * np.abs(bitrate_versions[bit_rate] -
                                                   bitrate_versions[last_bit_rate]) / M_IN_K
                r_batch.append(reward)

                last_bit_rate = bit_rate

                # log time_stamp, bit_rate, buffer_size, reward


                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                s_batch.append(state)

                for i in range(S_INFO):
                    for j in range(S_LEN):
                        log_file.write((str(state[i][j]) + '\t').encode())
                    log_file.write( ('\n' ).encode())
                log_file.write((str(bit_rate) + '\n').encode())
                log_file.flush()




                entropy_record.append(A_NN.compute_entropy(action_prob[0]))

                if end_of_video:
                    log_file.write('\n'.encode())
                    log_file.close()

                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)
                    entropy_record = []

                    print("video count", video_count)
                    video_count += 1

                    if video_count >= len(all_file_names):
                        break

                    log_path = dest_dir + '/'  + all_file_names[test_env.trace_idx]
                    log_file = open(log_path, 'wb')




    def _test(self, actor: A_NN.ActorNetwork, trace: AbrTrace,
              video_size_file_dir: str, save_dir: str, is_log: bool):
        os.makedirs(save_dir, exist_ok=True)
        if trace.name:
            log_name = os.path.join(save_dir, "{}.csv".format( trace.name))
        else:
            log_name = os.path.join(save_dir, "log.csv")
        abr_log = open(log_name, 'w')
        log_writer = csv.writer(abr_log, lineterminator='\n')
        log_writer.writerow(["timestamp", "bitrate", "buffer_size",
                             "rebuffering", "video_chunk_size", "delay",
                             "reward"])
        test_scheduler = TestScheduler(trace)
        net_env = Environment(test_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                              video_size_file_dir=video_size_file_dir)
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(self.a_dim)

        action_vec[bit_rate] = 1

        s_batch = [np.zeros((self.s_info, self.s_len))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        final_reward = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            (
                delay,
                sleep_time,
                buffer_size,
                rebuf,
                video_chunk_size,
                next_video_chunk_sizes,
                end_of_video,
                video_chunk_remain

            ) = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness

            reward = QoE_1(bit_rate, last_bit_rate, rebuf)

            r_batch.append(reward)

            last_bit_rate = bit_rate

            log_writer.writerow([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                                 buffer_size, rebuf, video_chunk_size, delay,
                                 reward])

            # log time_stamp, bit_rate, buffer_size, reward
            state = np.array(s_batch[-1], copy=True)
            # dequeue history record
            state = np.roll(state, -1, axis=1)
            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
                np.max(VIDEO_BIT_RATE)
            )  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = (
                    float(video_chunk_size) / float(delay) / M_IN_K
            )  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, : A_DIM] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
                                       )  # mega byte
            state[5, -1] = np.minimum(
                video_chunk_remain, TOTAL_VIDEO_CHUNK,
            ) / float(TOTAL_VIDEO_CHUNK)

            bit_rate, action_prob = self._get_next_bitrate(
                np.reshape(state, (1, self.s_info, self.s_len)),
                last_bit_rate, actor)

            s_batch.append(state)

            entropy_record.append(A_NN.compute_entropy(action_prob[0]))

            if end_of_video:

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                final_reward = sum(r_batch)
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(self.a_dim)

                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((self.s_info, self.s_len)))
                a_batch.append(action_vec)
                entropy_record = []

                break
        abr_log.close()

        return final_reward


    def train(self, args,trace_scheduler, val_traces: List[AbrTrace],
              save_dir: str, num_agents: int, total_epoch: int,
              video_size_file_dir: str, is_log: bool, model_save_interval: int ,is_master_train, entropy_weight,is_gener_2):

        assert self.train_mode

        # Visdom Settings
        # Visdom Logs
        val_epochs = []
        val_mean_rewards = []
        average_rewards = []
        average_entropies = []
        entropy_weight = entropy_weight
        actor_model_queue = deque(maxlen=QUEUE_LEN)

        logging.basicConfig(filename=os.path.join(save_dir, 'log_central'),
                            filemode='w', level=logging.INFO)

        # inter-process communication queues
        net_params_queues = []
        exp_queues = []
        for i in range(num_agents):
            net_params_queues.append(mp.Queue(1))
            exp_queues.append(mp.Queue(1))

        agents = []

        if is_gener_2:

            master_trace_path =  args.master_trace_path

            for i in range(num_agents):
                agents.append(mp.Process(
                    target=agent_gener_2,
                    args=(TRAIN_SEQ_LEN, self.s_info, self.s_len, self.a_dim,
                           i, net_params_queues[i], exp_queues[i], trace_scheduler,
                          video_size_file_dir, is_log, master_trace_path )))
        else:
            for i in range(num_agents):
                agents.append(mp.Process(
                    target=agent,
                    args=(TRAIN_SEQ_LEN, self.s_info, self.s_len, self.a_dim,
                          save_dir, i, net_params_queues[i], exp_queues[i], trace_scheduler,
                          video_size_file_dir, is_log)))
        for i in range(num_agents):
            agents[i].start()


        with tf.compat.v1.Session() as sess, \
                open(os.path.join(save_dir, 'log_train'), 'w', 1) as log_central_file, \
                open(os.path.join(save_dir, 'log_val'), 'w', 1) as val_log_file:
            log_writer = csv.writer(log_central_file, delimiter='\t', lineterminator='\n')
            log_writer.writerow(['epoch', 'loss', 'avg_reward', 'avg_entropy'])
            val_log_writer = csv.writer(val_log_file, delimiter='\t', lineterminator='\n')
            val_log_writer.writerow(
                ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
                 'rewards_median', 'rewards_95per', 'rewards_max'])

            epoch = 0

            actor = A_NN.ActorNetwork(sess,
                                     state_dim=[self.s_info, self.s_len],
                                     action_dim=self.a_dim,
                                     learning_rate = ACTOR_LR_RATE,
                                     bitrate_dim=A_DIM,
                                     entropy_weight=entropy_weight,
                                     name='actor'
                                             )

            critic1 = A_NN.CriticNetwork(sess,
                                       state_dim=[self.s_info, self.s_len],
                                       learning_rate=CRITIC_LR_RATE,
                                       bitrate_dim=A_DIM,
                                       name='QoE1')


            model_actor_old = A_NN.ActorNetwork(sess,
                                                       state_dim=[self.s_info, self.s_len],
                                                       action_dim=self.a_dim,
                                                       learning_rate=ACTOR_LR_RATE,
                                                       bitrate_dim=A_DIM,
                                                       entropy_weight=entropy_weight,
                                                       name='old_actor')

            logging.info('actor and critic initialized')

            sess.run(tf.compat.v1.global_variables_initializer())

            saver = tf.compat.v1.train.Saver(max_to_keep=None)  # save neural net parameters
            # restore neural net parameters
            if self.model_path:  # nn_model is the path to file
                saver.restore(sess, self.model_path)
                print("Model restored.")


            os.makedirs(os.path.join(save_dir, "model_saved"), exist_ok=True)

            if is_master_train:

                with tqdm(total=len(val_traces), desc="Validation Progress", unit="trace") as pbar:
                    val_rewards = []
                    for trace in val_traces:
                        reward = self._test(
                            actor, trace, video_size_file_dir=video_size_file_dir,
                            save_dir=os.path.join(save_dir, "val_logs"), is_log=is_log
                        )
                        val_rewards.append(reward)
                        pbar.update(1)


                val_log_writer.writerow(
                    [epoch, np.min(val_rewards),
                     np.percentile(val_rewards, 5), np.mean(val_rewards),
                     np.median(val_rewards), np.percentile(val_rewards, 95),
                     np.max(val_rewards)])

            po_buff = ReplayMemory(15 * TRAIN_SEQ_LEN)
            actor_val_queue = deque(maxlen = QUEUE_LEN)

            #-----------------------此为正常训练步骤----------------------------------------------------
            while epoch < total_epoch:

                if not is_master_train:
                    actor_model_queue.append(self.model_path)
                    break
                # synchronize the network parameters of work agent
                actor_net_params = actor.get_network_params()
                critic_net_params_1 = critic1.get_network_params()

                for i in range(num_agents):
                    net_params_queues[i].put([actor_net_params, critic_net_params_1])

                model_actor_old.set_network_params(actor_net_params)


                # 获取每个进程的轨迹s，a，r
                for i in range(num_agents):
                    # 将每个进程的数据暂时放入缓存区，需要时随机取出
                    if is_gener_2:
                        s_batch, a_batch,  Adv_batch_1, R_batch_1 ,master_s_batch, master_a_batch = exp_queues[i].get()

                        po_buff.push([s_batch, a_batch, Adv_batch_1,R_batch_1,master_s_batch, master_a_batch])

                    else:
                        s_batch, a_batch, Adv_batch_1, R_batch_1= exp_queues[i].get()
                        po_buff.push([s_batch, a_batch,  Adv_batch_1, R_batch_1])
                for _ in range(2):
                    #从缓冲区中随机取数据
                    if is_gener_2:
                        s_batch, a_batch,Adv_batch_1, R_batch_1,  master_s_batch_1, master_a_batch_1 = po_buff.sample(SAMPLE_LEN * 9)

                    else:
                        s_batch, a_batch,  Adv_batch_1,  R_batch_1  = po_buff.sample(SAMPLE_LEN * 18)


                    pro_old = model_actor_old.predict(s_batch)
                    if is_gener_2:
                        if epoch%2 ==0:
                            actor.train_2(s_batch, pro_old, a_batch,Adv_batch_1,
                                    master_s_batch_1,master_a_batch_1)
                        else:
                            actor.train_1(s_batch, pro_old, a_batch, Adv_batch_1)
                    else:
                        entropy_weight = actor.train_1(s_batch, pro_old, a_batch, Adv_batch_1)
                    critic1.train(s_batch, R_batch_1)





                #
                po_buff.clear()
                epoch += 1
                print(epoch)

                if epoch % model_save_interval == 0 and epoch!=0:
                    # # Visdom log and plot

                    with tqdm(total=len(val_traces), desc="Validation Progress", unit="trace") as pbar:
                        val_rewards = []
                        for trace in val_traces:
                            reward = self._test(
                                actor, trace, video_size_file_dir=video_size_file_dir,
                                save_dir=os.path.join(save_dir, "val_logs"), is_log=is_log
                            )
                            val_rewards.append(reward)
                            pbar.update(1)


                    # val_rewards = [self._test(
                    #     actor, trace, video_size_file_dir=video_size_file_dir,
                    #     save_dir=os.path.join(save_dir, "val_logs"), is_log=is_log) for trace in val_traces]
                    # val_mean_reward = np.mean(val_rewards)

                    val_log_writer.writerow(
                        [epoch, np.min(val_rewards),
                         np.percentile(val_rewards, 5), np.mean(val_rewards),
                         np.median(val_rewards), np.percentile(val_rewards, 95),
                         np.max(val_rewards)])
                    print("std:",np.std(val_rewards))

                    save_path = saver.save(
                        sess,
                        os.path.join(save_dir, "model_saved", f"nn_model_ep_{epoch}.ckpt"))
                    logging.info("Model saved in file: " + save_path)
                    #保存最新模型的位置
                    actor_model_queue.append(save_path)
                    # 将最新的模型的性能存入队列
                    actor_val_queue.append(np.mean(val_rewards))
                    # 判断是否达到初步训练的阈值 若达到阈值，则保存最后的模型路径，结束泛化者的初始训练
                    if not is_master_train:
                        if actor_val_queue[0] >= max(actor_val_queue)  and epoch > 40000:
                            break


        for tmp_agent in agents:
            tmp_agent.terminate()
            tmp_agent.join()


        return actor_model_queue[-1],entropy_weight


def standardize_advantage(advantage):
    positive_advantage = advantage[advantage > 0]
    negative_advantage = advantage[advantage < 0]
    zero_advantage = advantage[advantage == 0]

    if len(positive_advantage) > 0:
        min_positive = np.min(positive_advantage)
        max_positive = np.max(positive_advantage)

        standardized_positive = (positive_advantage - min_positive) / (max_positive - min_positive + 1e-8)  * 8 +1
    else:
        standardized_positive = np.array([])

    if len(negative_advantage) > 0:
        min_negative = np.min(negative_advantage)
        max_negative = np.max(negative_advantage)

        standardized_negative =  (negative_advantage - max_negative) / (max_negative - min_negative + 1e-8)  * 8 -1
    else:
        standardized_negative = np.array([])


    # 合并结果
    standardized_advantage = np.zeros_like(advantage)
    standardized_advantage[advantage > 0] = standardized_positive
    standardized_advantage[advantage < 0] = standardized_negative
    standardized_advantage[advantage == 0] = zero_advantage


    return standardized_advantage




def agent(train_seq_len: int, s_info: int, s_len: int, a_dim: int,
          save_dir: str, agent_id: int, net_params_queue: mp.Queue,
          exp_queue: mp.Queue, trace_scheduler, video_size_file_dir: str,
          is_log: bool):
    """Agent method for A2C/A3C training framework.

    Args
        train_seq_len: train batch size
        net_params_queues: a queue for the transferring neural network
                           parameters between central agent and agent.
        exp_queues: a queue for the transferring experience/rollouts
                    between central agent and agent.
    """
    # if agent_id < 5:
    #     gpu_id = 0
    # else:
    #     gpu_id = 1
    #
    # gpu_id = 0
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
    #     tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    #     print(f"Agent {agent_id}: Using GPU {gpu_id}")
    # else:
    #     print(f"Agent {agent_id}: No GPU devices found.")


    net_env = Environment(trace_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                          video_size_file_dir=video_size_file_dir,
                          random_seed=agent_id)

    with tf.compat.v1.Session() as sess:



        actor = A_NN.ActorNetwork(sess, state_dim=[s_info, s_len],
                                 action_dim=a_dim, learning_rate = ACTOR_LR_RATE,bitrate_dim=A_DIM,entropy_weight=np.log(A_DIM),
                                         name = 'actor')
        # learning_rate=args.ACTOR_LR_RATE)
        critic1 = A_NN.CriticNetwork(sess,
                                                 state_dim=[s_info, s_len],
                                                 learning_rate=CRITIC_LR_RATE,
                                                 bitrate_dim=A_DIM,
                                                 name='QoE1')


        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params1= net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic1.set_network_params(critic_net_params1)



        last_bit_rate = DEFAULT_QUALITY
        selection = 0
        bit_rate = DEFAULT_QUALITY

        # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
        action_vec = np.zeros(a_dim)

        action_vec[bit_rate] = 1

        s_batch = [np.zeros((s_info, s_len))]
        a_batch = [action_vec]
        r_batch_1 = []



        entropy_record = []


        epoch = 0
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            if is_log:
                reward_1 = QoE_1(bit_rate, last_bit_rate, rebuf)

            else:
                reward_1 = QoE_1(bit_rate, last_bit_rate, rebuf)


            r_batch_1.append(reward_1)


            last_bit_rate = bit_rate

            # retrieve previous state
            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be args.S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                           float(np.max(VIDEO_BIT_RATE))  # last quality

            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                           float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, TOTAL_VIDEO_CHUNK) / float(TOTAL_VIDEO_CHUNK)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(
                state, (1, s_info, s_len)))


            action_cumsum = np.cumsum(action_prob)

            bit_rate = (
                    action_cumsum
                    > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
            ).argmax()

            # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(A_NN.compute_entropy(action_prob[0]))

            # report experience to the coordinator
            if len(r_batch_1) >= train_seq_len or end_of_video:

                s_batch_np = np.stack(s_batch, axis=0)
                a_batch_np = np.vstack(a_batch)
                r_batch_np_1 = np.vstack(r_batch_1)




                assert s_batch_np.shape[0] == a_batch_np.shape[0]
                assert s_batch_np.shape[0] == r_batch_np_1.shape[0]
                ba_size = s_batch_np.shape[0]
                v_batch_1 = critic1.predict(s_batch_np)
                R_batch_1 = np.zeros(r_batch_np_1.shape)

                if end_of_video:
                    R_batch_1[-1, 0] = 0  # terminal state
                else:
                    R_batch_1[-1, 0] = v_batch_1[-1, 0]  # boot strap from last state

                for t in reversed(range(ba_size - 1)):
                    R_batch_1[t, 0] = r_batch_np_1[t] + GAMMA * R_batch_1[t + 1, 0]

                Adv_batch_1 = R_batch_1 - v_batch_1

                Adv_batch_1_standardized = standardize_advantage(Adv_batch_1)

                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               Adv_batch_1_standardized[1:],
                               R_batch_1[1:],
                               ])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params_1= net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic1.set_network_params(critic_net_params_1)

                del s_batch[:]
                del a_batch[:]
                del r_batch_1[:]

                del entropy_record[:]

                # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros(a_dim)

                action_vec[bit_rate] = 1
                s_batch.append(np.zeros((s_info, s_len)))
                a_batch.append(action_vec)
                epoch += 1
                net_env.trace_scheduler.set_epoch(epoch)

            else:
                s_batch.append(state)
                # print(bit_rate)
                # action_vec = np.zeros(args.A_DIM)
                # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros(a_dim)

                action_vec[bit_rate] = 1
                # print(action_vec)
                a_batch.append(action_vec)


def agent_gener_2(train_seq_len: int, s_info: int, s_len: int, a_dim: int,
                    agent_id: int, net_params_queue: mp.Queue,
                    exp_queue: mp.Queue, trace_scheduler, video_size_file_dir: str,
                    is_log: bool,master_trace_dir:str):


    if agent_id < 5:
        gpu_id = 0
    else:
        gpu_id = 1
    gpu_id = 0
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
        print(f"Agent {agent_id}: Using GPU {gpu_id}")
    else:
        print(f"Agent {agent_id}: No GPU devices found.")

    print("master_trace:",master_trace_dir)

    net_env = Environment(trace_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                          video_size_file_dir=video_size_file_dir,
                          random_seed=agent_id)


    with tf.compat.v1.Session() as sess:

        actor = A_NN.ActorNetwork(sess, state_dim=[s_info, s_len],
                                              action_dim=a_dim, learning_rate=ACTOR_LR_RATE, bitrate_dim=A_DIM,
                                              entropy_weight=np.log(A_DIM),
                                              name='actor')
        # learning_rate=args.ACTOR_LR_RATE)
        critic1 = A_NN.CriticNetwork(sess,
                                                 state_dim=[s_info, s_len],
                                                 learning_rate=CRITIC_LR_RATE,
                                                 bitrate_dim=A_DIM,
                                                 name='QoE1')


        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params1 = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic1.set_network_params(critic_net_params1)


        last_bit_rate = DEFAULT_QUALITY
        selection = 0
        bit_rate = DEFAULT_QUALITY

        # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
        action_vec = np.zeros(a_dim)

        action_vec[bit_rate] = 1

        s_batch = [np.zeros((s_info, s_len))]
        a_batch = [action_vec]
        r_batch_1 = []



        entropy_record = []
        epoch = 0


        file_name = net_env.file_name



        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real


            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            reward_1 = QoE_1(bit_rate, last_bit_rate, rebuf)

            r_batch_1.append(reward_1)


            last_bit_rate = bit_rate

            # retrieve previous state
            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be args.S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                           float(np.max(VIDEO_BIT_RATE))  # last quality

            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                           float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, TOTAL_VIDEO_CHUNK) / float(TOTAL_VIDEO_CHUNK)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(
                state, (1, s_info, s_len)))


            action_cumsum = np.cumsum(action_prob)

            bit_rate = (
                    action_cumsum
                    > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
            ).argmax()

            # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(A_NN.compute_entropy(action_prob[0]))

            # report experience to the coordinator
            if len(r_batch_1) >= train_seq_len or end_of_video:

                master_s_batch = []
                master_a_batch = []



                master_trace_file = master_trace_dir  + '/' + str(file_name)
                with open(master_trace_file, 'r') as file:
                    lines = file.readlines()
                for i in range(0, len(lines), 7):

                    if i + 6 < len(lines)-7:
                        temp_data = np.zeros((s_info, s_len ))
                        for j in range(6):
                            line_data = [float(x) for x in lines[i + j].strip().split('\t')]
                            temp_data[j] = line_data

                        # 获取action
                        action_data = int(lines[i+6])
                        master_action_vec = np.zeros(a_dim)

                        master_action_vec[action_data] = 1

                        # 将数据添加到 a_batch、s_batch 中
                        master_a_batch.append(master_action_vec)
                        master_s_batch.append(temp_data)



                s_batch_np = np.stack(s_batch, axis=0)
                a_batch_np = np.vstack(a_batch)
                r_batch_np_1 = np.vstack(r_batch_1)

                assert s_batch_np.shape[0] == a_batch_np.shape[0]
                assert s_batch_np.shape[0] == r_batch_np_1.shape[0]
                ba_size = s_batch_np.shape[0]
                v_batch_1 = critic1.predict(s_batch_np)

                R_batch_1 = np.zeros(r_batch_np_1.shape)
                if end_of_video:
                    R_batch_1[-1, 0] = 0  # terminal state
                else:
                    R_batch_1[-1, 0] = v_batch_1[-1, 0]  # boot strap from last state

                for t in reversed(range(ba_size - 1)):
                    R_batch_1[t, 0] = r_batch_np_1[t] + GAMMA * R_batch_1[t + 1, 0]


                Adv_batch_1 = R_batch_1 - v_batch_1
                Adv_batch_1_standardized = standardize_advantage(Adv_batch_1)

                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               Adv_batch_1_standardized[1:],
                               R_batch_1[1:],
                               master_s_batch,
                               master_a_batch,
                               ])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params_1= net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic1.set_network_params(critic_net_params_1)


                del s_batch[:]
                del a_batch[:]
                del r_batch_1[:]

                del entropy_record[:]


            # store the state and action into batches
            if end_of_video:
                epoch += 1

                file_name = net_env.file_name
                net_env.trace_scheduler.set_epoch(epoch)

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros(a_dim)

                action_vec[bit_rate] = 1
                s_batch.append(np.zeros((s_info, s_len)))
                a_batch.append(action_vec)



            else:
                s_batch.append(state)
                action_vec = np.zeros(a_dim)

                action_vec[bit_rate] = 1
                # print(action_vec)
                a_batch.append(action_vec)





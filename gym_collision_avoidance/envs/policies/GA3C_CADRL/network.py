import numpy as np
import tensorflow as tf
import time

np.set_printoptions(precision=3, suppress=True)

class Actions():
    # Define 11 choices of actions to be:
    # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
    # [0.5*v_pref,  [-pi/6, 0, pi/6]]
    # [0,           [-pi/6, 0, pi/6]]
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

class NetworkVPCore(object):
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

    def crop_x(self, x):
        # stupid stuff because each NN might accept diff length observation
        # if not hasattr(self, 'x'):
        #     return x
        if x.shape[-1] > self.x.shape[-1]:
            x_ = x[:,:self.x.shape[-1]]
        elif x.shape[-1] < self.x.shape[-1]:
            x_ = np.zeros((x.shape[0], self.x.shape[-1]))
            x_[:,:x.shape[1]] = x
        else:
            x_ = x
        return x_

    def predict_p(self, x):
        x = self.crop_x(x)
        # print(x)
        # assert(0)
        return self.sess.run(self.softmax_p, feed_dict={self.x: x})

    def simple_load(self, filename=None):
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
            raise NotImplementedError
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))

                new_saver = tf.train.import_meta_graph(filename+'.meta', clear_devices=True)
                self.sess.run(tf.global_variables_initializer())
                new_saver.restore(self.sess, filename)

                # for n in tf.get_default_graph().as_graph_def().node:
                #     print(n)
                # assert(0)

                # all_ops = tf.get_default_graph().get_operations()
                # for op in all_ops:
                #     print(op)

                # assert(0)

                self.softmax_p = g.get_tensor_by_name('Softmax:0')
                self.x = g.get_tensor_by_name('X:0')
                self.v = g.get_tensor_by_name('Squeeze:0')

class NetworkVP_rnn(NetworkVPCore):
    def __init__(self, device, model_name, num_actions):
        super(self.__class__, self).__init__(device, model_name, num_actions)

if __name__ == '__main__':
    actions = Actions().actions
    num_actions = Actions().num_actions
    nn = NetworkVP_rnn("/cpu:0", 'network', num_actions)
    nn.simple_load()
    assert(0)



    # unlikely to still work right off bat
    actions = Actions().actions
    num_actions = Actions().num_actions
    nn = NetworkVP_rnn(Config.DEVICE, 'network', num_actions)
    nn.simple_load()

    obs = np.zeros((Config.FULL_STATE_LENGTH))
    obs = np.expand_dims(obs, axis=0)

    num_queries = 10000
    t_start = time.time()
    for i in range(num_queries):
        obs[0,0] = 10 # num other agents
        obs[0,1] = np.random.uniform(0.5, 10.0) # dist to goal
        obs[0,2] = np.random.uniform(-np.pi, np.pi) # heading to goal
        obs[0,3] = np.random.uniform(0.2, 2.0) # pref speed
        obs[0,4] = np.random.uniform(0.2, 1.5) # radius
        predictions = nn.predict_p(obs)[0]
    t_end = time.time()
    print("avg query time:", (t_end - t_start)/num_queries)
    print("total time:", t_end - t_start)
    # action = actions[np.argmax(predictions)]
    # print "action:", action

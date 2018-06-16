import numpy as np
import matplotlib.pyplot as plt

class QLearnAgent:
    
    def __init__(self, maze_matrix, action_array, r_goal):
       
        self.maze = maze_matrix
        S, R, m = self.__transform_maze_matrix(maze_matrix)
        self.S = S
        self.R = R
        self.m = m
        self.A = self.__validate_actions(self.S, action_array)
       
        self.r_goal = r_goal
        
            
    def __transform_maze_matrix(self, maze_matrix):
        """
        Utility to transform matrix into S and R.
        """
        nx, ny = maze_matrix.shape
        m = 10**int(np.ceil(np.log10(ny)))

        S = []
        R = []

        for i in range(nx):
            for j in range(ny):
                state = maze_matrix[i,j]
                R.append(state)
                S.append(m*j + i)

        S = np.array(S).astype(int)
        R = np.array(R).astype(float)
        
        return S, R, m

    def __validate_actions(self, S, A):

        S = np.asarray(S)
        A = np.asarray(A)

        na = len(A)
        ns = len(S)

        A1 = np.zeros([ns, na])

        for i,s in enumerate(S):
            for j,a in enumerate(A):
                s1 = self.action(s, a)
                if s1 in S:
                    A1[i,j] = A[j]
                else:
                    A1[i,j] = np.nan

        A1 = np.ma.masked_invalid(A1).astype(int)
        return A1    

    def __xyflat_to_xypairs(self, xyflat, m=10):
        xyflat = np.array(xyflat)
        y = xyflat // m
        x = np.mod(xyflat, m)
        return x, y    
    
    def action(self, s, a):
        return s + a    
    
    def train(self, nmax_episodes=100, alpha=0.3, gamma=0.6, Q_init=None,
              l1=100, l2=40, pmax=0.3, pmin=0.02, random_state=666):
        
        self.nmax_episodes = nmax_episodes
        self.alpha = alpha
        self.gamma = gamma
        
        assert pmax <= 1., '0 <= pmin < pmax <= 1'
        assert pmin < pmax, '0 <= pmin < pmax <= 1'
        assert pmin >= 0., '0 <= pmin < pmax <= 1'
        
        if Q_init is None:
            np.random.seed(random_state)
            self.Q = np.random.rand(self.A.shape[0], self.A.shape[1])
        else:
            assert np.shape(Q_init) == (self.A.shape[0], self.A.shape[1]), 'Q shape must be "(len(S), len(A))"'
            self.Q = np.array(Q_init)

        self.Q = np.ma.masked_where(self.A.mask, self.Q)
        self.Q_init = self.Q.copy()       
        self.err_q = np.zeros(self.nmax_episodes)
       
        self.n_iter = 0
        self.n_steps = 0
        self.n_episodes = 0
        self.s = np.random.permutation(self.S).take(0)
        
        while self.n_episodes < self.nmax_episodes:
            
            # index of the current state
            i_s = np.where(self.S==self.s)[0].take(0)
    
            # random exploration/exploitation
            #pmax, pmin, l1, l2
            eps1 = np.clip( pmax * np.exp(-self.n_episodes / l1), pmin, 1. )
            eps2 = np.clip( pmax * (1. - np.exp(-(self.n_steps / l2))), pmin, 1. )
            
            if np.random.rand() < max(eps1, eps2):
                a = np.random.permutation(self.A[i_s][self.A[i_s].mask == False])[0].take(0)
                i_a = np.where(self.A[i_s] == a)[0].take(0)              
                #print('random', eps1, eps2, self.n_steps)

            else:
                i_a = np.argmax(self.Q[i_s,:]).take(0)
                a = self.A[i_s, i_a].take(0)

            # imediate reward for taking action a at state s
            r = self.R[i_s]
            
            self.n_steps += 1
            self.n_iter += 1
            
            if r == self.r_goal:
                
                self.err_q[self.n_episodes] = np.sum((self.Q_init - self.Q)**2)
                # estimate the thorugh Q_star instead
                
                self.n_episodes += 1
                self.n_steps = 0
                self.s = np.random.permutation(self.S).take(0)
                
                
            else:
                # the new state after taking action a at state s
                s1 = self.action(self.s, a)

                # future reward
                i_s1 = np.where(self.S==s1)[0].take(0)
                i_a1 = np.asarray(np.argmax(self.Q[i_s1,:])).take(0)

                # updating the Q table
                self.Q[i_s,i_a] = (1.-self.alpha)*self.Q[i_s,i_a] + self.alpha*(r + self.gamma*self.Q[i_s1,i_a1])

                # saving the new state
                self.s = s1
                
            
        self.Q_star = np.argmax(self.Q, axis=1)
        self.A_star = np.array([self.A[i,iq] for (i,iq) in enumerate(self.Q_star)]).astype(int)        
        self.err_q = self.err_q[0] / self.err_q
        
    def play(self, s, max_episodes=100, verbose=True):
        
        i_s = np.where(self.S == s)[0].take(0) # 0
        
        #a = self.A[self.Q_star[i_s]]
        a = self.A_star[i_s]
        r = self.R[i_s]
        
        slist = []
        alist = []
        rlist = []
        
        slist.append(s)
        alist.append(a)
        rlist.append(r)

        i = 0
        if verbose:
            print('i: {:03d}, s: {:02d}, a: {:02d}, r: {:5.1f}, i_s: {:03d}'.format(i, s, a, r, i_s))
        
        while r != self.r_goal and i < max_episodes:
            
            s = self.action(s, a)
            
            i_s = np.where(self.S == s)[0].take(0)
            
            #a = self.A[self.Q_star[i_s]]
            a = self.A_star[i_s]
            r = self.R[i_s]
                   
            slist.append(s)
            alist.append(a)
            rlist.append(r)
            
            i += 1
            if verbose:           
                print('i: {:03d}, s: {:02d}, a: {:02d}, r: {:5.1f}, i_s: {:03d}'.format(i, s, a, r, i_s))
            
        s_arr = np.array(slist)
        a_arr = np.array(alist)
        r_arr = np.array(rlist)
        
        if verbose:
            print('\n  Total points: {}\n'.format(r_arr.sum()))
        
        return s_arr, a_arr, r_arr
    
    def plot(self, s, max_episodes=100, verbose=False):
        
        s_arr, a_arr, r_arr = self.play(s, max_episodes, verbose=verbose)
        x, y = self.__xyflat_to_xypairs(xyflat=s_arr, m=self.m)
        
        vmax = np.abs(self.R).max()
        vmin = -vmax

        plt.clf()
        plt.pcolor(self.maze.T, cmap=plt.cm.RdYlBu, vmin=vmin, vmax=vmax)
        plt.grid()
        plt.plot(x[:]+0.5, y[:]+0.5, 'k-o')
        plt.plot(x[0]+0.5, y[0]+0.5, 'r*', ms=16)
        plt.plot(x[-1]+0.5, y[-1]+0.5, 'ro', ms=12)
        plt.show()
    
    def multi_plot(self, slist, basename='_fig'):

        vmax = np.abs(self.R).max()
        vmin = -vmax

        ls = int(1 + np.ceil(np.log10(self.maze.shape[0] * self.maze.shape[1])))
        le = int(np.ceil(np.log10(len(slist))))

        for k, s in enumerate(slist):

            s_arr, a_arr, r_arr = self.play(s, verbose=False)
            x, y = self.__xyflat_to_xypairs(xyflat=s_arr, m=self.m)

            ns = len(s_arr)

            episode = str(k+1).zfill(le)

            fig = plt.figure()

            for j in range(1, ns+1):

                step = str(j).zfill(ls)

                fname = '{}_episode{}_step{}.png'.format(basename, episode, step)
                print(fname)

                fig.clf()            
                ax = fig.add_subplot(111)
                ax.pcolor(self.maze.T, cmap=plt.cm.RdYlBu, vmin=vmin, vmax=vmax)
                ax.plot(x[:j]+0.5, y[:j]+0.5, 'k-o')        
                ax.plot(x[0]+0.5, y[0]+0.5, 'r*', ms=15)

                if j == ns:  
                    ax.plot(x[-1]+0.5, y[-1]+0.5, 'rH', ms=15)

                ax.grid()

                fig.savefig(fname, dpi=100, bbox_inches='tight')

            plt.close('all')    

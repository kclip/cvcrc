
## Import modules

import torch
import torch.nn as nn
import numpy as np
import datetime
import argparse
from scipy.io import loadmat, savemat
from copy import deepcopy


class cumulative_hazard(nn.Module): # f1: RNN = recurrent neural network --> f2: MLP for the cumulative hazard function
    def __init__(   self,
                    lookahead =         12,
                    f1_num_neurons =    8,      # number of neurons in each layer of f1
                    f2_num_layers =     3,      # num of layers
                    f2_num_neurons =    8  ):   # number of neurons in the each of the layers of f2
        super(cumulative_hazard, self).__init__()
        self.lookahead =                lookahead
        self.f1_num_neurons =           f1_num_neurons
        self.f2_num_layers =            f2_num_layers
        self.f2_num_neurons =           f2_num_neurons

        ######### network f1 (RNN) ##########
        setattr(self, 'f1',         nn.Linear(      in_features =    int(f1_num_neurons + 1), # the 1 stands for the recurrency of x
                                                    out_features =   int(f1_num_neurons),
                                                    bias =           True)  )
        self.active_f1 =            torch.tanh

        ######### network f2 (MLP) ##########
        setattr(self, 'f2_tau',     nn.Linear(      in_features=    int(1),  # the 1 stands for the recurrency of x
                                                    out_features=   int(f2_num_neurons),
                                                    bias =          True))
        v_f2_layers =               [f1_num_neurons] + (f2_num_layers-1) * [f2_num_neurons] + [1]
        for ind_layer in range(len(v_f2_layers)-1):
            cur_fc_name =          'f2_' + str(ind_layer)
            setattr(self, cur_fc_name, nn.Linear(   in_features =   int(v_f2_layers[ind_layer]),
                                                    out_features =  int(v_f2_layers[ind_layer+1]),
                                                    bias =          True)  )
        self.active_f2_hidden =      torch.tanh
        self.active_f2_output =      nn.functional.softplus

    def forward(self,
                Y_seq,              # a batch of sequences of point processes
                tau_vec             # second input of the MLP
                ):                  # returns median values of logits of num_cat, without the torch.nn.functional.softmax
        ## f1 ##
        assert                      (Y_seq.shape[1] > self.lookahead),(f'Y_seq.shape[1] = {Y_seq.shape[1]} (should be >) {self.lookahead} = self.lookahead')
        Y_seq_diff =                torch.log(Y_seq[:,-self.lookahead-1:].diff(dim=1)+0.1) # the 0.1 is for numerical reasons, avoiding log(0)
        flow =                      torch.zeros(Y_seq.shape[0],self.f1_num_neurons)
        for ind_layer in range(self.lookahead-1):
            flow =                  torch.cat(tensors=(flow,Y_seq_diff[:,ind_layer].view(-1,1)),dim=1)
            flow =                  self.active_f1(self.f1(flow)) # linear with activation
        ## f2 ##
        for ind_layer in range(self.f2_num_layers):
            cur_fc_name =          'f2_' + str(ind_layer)
            if ind_layer==0:
                flow =              torch.nn.functional.linear(     flow,
                                                                    self.f2_0.weight,
                                                                    self.f2_0.bias       )
                flow +=             torch.nn.functional.linear(     tau_vec,  # making sure all weights and biases are non negative
                                                                    torch.log(1+torch.exp(self.f2_tau.weight)),
                                                                    torch.log(1+torch.exp(self.f2_tau.bias)  )    )
            else:
                flow =              torch.nn.functional.linear(     flow,  # making sure all weights and biases are non negative
                                                                    torch.log(1+torch.exp(getattr(self,cur_fc_name).weight)),
                                                                    torch.log(1+torch.exp(getattr(self,cur_fc_name).bias  ))   )
            if (ind_layer < self.f2_num_layers-1): # apply activation to all but last
                flow =              self.active_f2_hidden(flow)
        flow =                      self.active_f2_output(flow)
        return flow     # returns a vector of activated outputs


    def loss_of_batch(self, Y_seq):
        tau_mat =                   Y_seq.diff(dim=1)
        N,seq_len =                 Y_seq.shape
        loss =                      0.0
        for ii in range(self.lookahead,seq_len-1):
            tau_ii =                tau_mat[:,ii].view(N,1).detach().clone().requires_grad_(True)
            Phi_ii =                self(Y_seq[:,:ii+1],tau_ii) # forward pass
            grad_Phi_ii =           torch.autograd.grad(    Phi_ii,
                                                            tau_ii,
                                                            create_graph=True,
                                                            grad_outputs=torch.ones_like(tau_ii))[0]
            loss +=                 - (torch.log(torch.max(grad_Phi_ii, torch.tensor(1e-10))) - Phi_ii ).sum() # sum here, norm into mean in next line
        loss /=                     (seq_len-1-self.lookahead) * N
        return loss


    def predict_ahead(self, X_seq, m):
        N, x_seq_len =              X_seq.shape
        Y_aug =                     torch.cat(tensors=(X_seq,torch.zeros(N,m)),dim=1)
        for ii in range(x_seq_len, x_seq_len + m):
            l =                     0.0     + torch.zeros(N,1)                  # left  for bisection, for each point. dim = [N,1]
            r =                     50000.0 + torch.zeros(N,1)                  # right for bisection, for each point. dim = [N,1]
            for jj in range(24):
                c =                 (l + r) / 2                                 # dim = [N,1]
                v =                 self(Y_aug[:,:ii], c) - np.log(2)           # forward pass
                l[v <  0] =         c[v <  0]
                r[v >= 0] =         c[v >= 0]
            Y_aug[:,ii] =           ((l + r) / 2).view(-1) + Y_aug[:,ii-1]      # outcome of bisection
        Y_pred =                    Y_aug[:,-m:]                                # last m items of the augmented are the predicted ones
        return Y_pred


def train_net(net, Y_seq, lr, num_iters, weight_decay):
    optimizer =                     torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for ii in range(num_iters):
        optimizer.zero_grad()
        loss =                      net.loss_of_batch(Y_seq)
        loss.backward()
        optimizer.step()
        if (ii%10==1 or ii==num_iters-1):
            elapsed_time(f'-----> {ii}/{num_iters} training loss={loss.item()}')

def Gamma_as_intervals_1st_part(X,phis,m):
    # size(X)                   [ N, d     ]
    # size(phis)                [ K        ]
    # size(Yhat)                [ N, m, K  ]
    # m                         number of predictions ahead
    N =                         X.shape[0]
    K =                         len(phis)
    Yhat =                      torch.zeros(N, m, K)
    for k in range(K):
        with torch.no_grad():
            Yhat[:,:,k] =       phis[k].predict_ahead(X, m)
    return Yhat

def Gamma_as_intervals_2nd_part(Yhat,lambda_vec,interval_common_ratio):
    # size(Yhat)                [ N, m, K  ]  Only one interval
    # size(lambda_vec)          [len_lambda]
    # size(Gamma)               [ N, m, 2, K , len_lambda]  Only one interval
    N,m,K =                     Yhat.shape
    len_lambda =                len(lambda_vec)
    geo_seq =                   interval_common_ratio**torch.arange(m) # size = [m]
    Gamma =                     torch.zeros(N, m, 2, K, len_lambda)
    Gamma[:,:,0,:,:] =          Yhat.view(N,m,K,1) - geo_seq.view(1,m,1,1) * lambda_vec.view(1,1,1,-1) /2 # [ N, m, K  ]  -> [ N, m, K, 1  ]   and [len_lambda] -> [1,1,1,len_lambda]
    Gamma[:,:,1,:,:] =          Yhat.view(N,m,K,1) + geo_seq.view(1,m,1,1) * lambda_vec.view(1,1,1,-1) /2
    return Gamma

def Gamma_as_intervals(X, phis, lambda_scalar, m, interval_common_ratio):
    # size(X)                   [ N, d        ]
    # size(phis)                [ K           ]
    # size(lambda_scalar)       [             ]
    # size(Gamma)               [ N, m, 2, K  ]  Only one interval
    # m                         number of predictions ahead
    N =                         X.shape[0]
    K =                         len(phis)
    geo_seq =                   interval_common_ratio ** torch.arange(m)  # size = [m]
    Gamma =                     torch.zeros(N, m, 2, K)
    for k in range(K):
        with torch.no_grad():
            Yhat =          phis[k].predict_ahead(X, m)
        Gamma[:,:,0,k] =    Yhat - geo_seq.view(1,m) * lambda_scalar/2  # lambda is the uncertainty window, so this is the inverse
        Gamma[:,:,1,k] =    Yhat + geo_seq.view(1,m) * lambda_scalar/2
    return Gamma


def risk_ell(Y, Gamma):
    # size(Y)       [ N , m      ]
    # size(Gamma)   [ N , m, 2*K ]
    # size(risk)    [ N          ]
    N,m =                               Y.shape
    K =                                 int(Gamma.shape[2]/2)
    is_y_in_Gamma =                     torch.zeros(N,m,K)
    for k in range(K):
        is_y_in_Gamma[:,:,k] =  ( (Y >= Gamma[:,:,2*k]) & (Y <= Gamma[:,:,2*k+1]) ) # element-wise logical and
    risk =                              is_y_in_Gamma.any(dim=2).logical_not().float().mean(dim=1)
    return risk

def dim_wise_union(GammaIn):
    # size(GammaIn)             [ N , m, 2  , K ]
    # size(GammaOut)            [ N , m, 2*K    ] # null intervals will be located at the end with [inf,inf] represenation
    (N,m,_,K) =                 GammaIn.shape
    GammaOut =                  float('inf')*torch.ones(N,m,2*K)
    for n in range(N):
        for j in range(m):
            borders=            (GammaIn[n,j,:,:]).flatten().unique().sort().values
            representors =      0.5*(   torch.cat(tensors=(torch.tensor([-float('inf')]),borders),dim=0)
                                      + torch.cat(tensors=(borders,torch.tensor([float('inf')])),dim=0))
            mask =              torch.zeros_like(representors,dtype=torch.bool)
            for k in range(K):
                mask =          mask | (  (representors >= GammaIn[n,j,0,k]) & (representors <= GammaIn[n,j,1,k] )  )
            diff_mask =         torch.diff(mask.float())
            risings =           borders[diff_mask==+1]
            fallings =          borders[diff_mask==-1] # length should be the same as risings
            GammaOut[n,j,0:2*len(risings ):2] =    risings
            GammaOut[n,j,1:2*len(fallings):2] =    fallings
    return GammaOut

def calc_ineff_of_intervals(Gamma): # mean over samples of the log of the d-dim volume
    # size(Gamma)       [ N , m, 2*K    ]
    # size(ineff        [ N , m         ]
    ineff =         (Gamma[:,:,1::2]-Gamma[:,:,0::2]).nansum(dim=2)#.mean(dim=1) # omit NaN for the unused intervals which are inf-inf
    return ineff

def generate_hawkes2(N,N_sam_in_seq): # based on https://github.com/omitakahiro/NeuralNetworkPointProcess
    Y_seq_tot =                                     torch.zeros((N,N_sam_in_seq),dtype=torch.float32)
    for ii in range(N):
        Y_seq_tot[ii,:] =                           simulate_hawkes(N_sam_in_seq ,0.2 ,[0.4 ,0.4] ,[1.0 ,20.0])
    return Y_seq_tot

def simulate_hawkes(n ,mu ,alpha ,beta):  # code source: https://github.com/omitakahiro/NeuralNetworkPointProcess
    T = []
    #LL = []
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential( ) /l
        x = x + step
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0 ] *step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1 ] *step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0 ] *step)
        l_trg2 *= np.exp(-beta[1 ] *step)
        l_next = mu + l_trg1 + l_trg2
        if np.random.rand() < l_next /l:  # accept
            T.append(x)
            #LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0 ] *beta[0]
            l_trg2 += alpha[1 ] *beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            if count == n:
                break
    return torch.tensor(T) #[np.array(T) ,np.array(LL)]

def elapsed_time(msg):
    print(f"Elapsed time {datetime.datetime.now() - startTime}: " + msg)

########################################################################################################################
########################################################################################################################
########################################################################################################################
## main begins here:
########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == "__main__":
    print(f'{datetime.datetime.now()}: First line of main')
    startTime =                         datetime.datetime.now()
    parser =                            argparse.ArgumentParser(description='Conformal Set Prediction')

    parser.add_argument(    '--N_av',                   dest='N_av',                    default=50,     type=int)      # batch size of available sequences
    parser.add_argument(    '--N_te',                   dest='N_te',                    default=1000,   type=int)      # batch size of test sequences
    parser.add_argument(    '--seed',                   dest='seed',                    default=16,     type=int)      # random seed
    parser.add_argument(    '--is_data_synthetic',      dest='is_data_synthetic',       default=1,      type=int)
    parser.add_argument(    '--alpha',                  dest='alpha',                   default=0.17,   type=float)
    parser.add_argument(    '--lr',                     dest='lr',                      default=0.05,   type=float)
    parser.add_argument(    '--weight_decay',           dest='weight_decay',            default=0.0002, type=float)
    parser.add_argument(    '--interval_common_ratio',  dest='interval_common_ratio',   default=1.2,    type=float)    # the m intervals are a geometric seq with this common ratio
    parser.add_argument(    '--num_iters',              dest='num_iters',               default=500,    type=int)      # number of training iterations
    parser.add_argument(    '--num_predictions_ahead',  dest='num_predictions_ahead',   default=6,      type=int)      # lookahead range, used for the risk
    parser.add_argument(    '--num_folds',              dest='num_folds',               default=12,     type=int)      # K, number of folds in K-CV-CRC
    parser.add_argument(    '--N_sam_in_seq',           dest='N_sam_in_seq',            default=60,     type=int)      # total length inm smaples of each point process

    args =                                  parser.parse_args()
    seed=                                   args.seed
    N_av =                                  args.N_av
    print(f'args detected N_av={N_av} and seed={seed}')
    N_te =                                  args.N_te
    torch.manual_seed(                      seed)
    N_sam_in_seq =                          args.N_sam_in_seq
    if args.is_data_synthetic:
        Y_seq_tot =                         generate_hawkes2(N_av+N_te, N_sam_in_seq)
    else:
        lm =                                loadmat('twitter_20000_random_seqs.mat')
        N_seq_as_in_file =                  lm['N_seq'].item()
        seq_len_as_in_file =                lm['seq_len'].item()
        Y_seq_tot =                         torch.tensor(lm['Y_seq_tot'],dtype=torch.float32) # total sequances
        # shuffle sequences randomly, and focus on shorter sequences, for faster run-time
        Y_seq_tot =                         Y_seq_tot[torch.randperm(Y_seq_tot.shape[0]),:N_sam_in_seq]
    lr =                                    args.lr
    weight_decay =                          args.weight_decay
    num_iters =                             args.num_iters
    m =                                     args.num_predictions_ahead
    interval_common_ratio =                 args.interval_common_ratio
    B =                                     1.0           # uppper bound of the risk
    alpha =                                 args.alpha
    lambda_vec =                            torch.logspace(0,6,481)
    len_lambda =                            len(lambda_vec)
    Y_seq_av =                              Y_seq_tot[      : N_av, :]
    Y_seq_te =                              Y_seq_tot[-N_te :     , :]
    y_seq_te =                              Y_seq_te[0]
    init_net =                              cumulative_hazard()


    ##   VB-CRC
    N_tr =                                  int(N_av*0.5) # num of proper training points
    N_val =                                 N_av - N_tr
    assert(                                 N_val >= 1/alpha )
    Y_seq_vb__tr =                          Y_seq_av[        : N_tr           , : ]
    Y_seq_vb__val =                         Y_seq_av[ N_tr   : N_tr + N_val : , : ]
    phi_vb_ =                               deepcopy(init_net)
    elapsed_time('VB before training')
    train_net(                              phi_vb_, Y_seq_vb__tr, lr=lr, num_iters=num_iters, weight_decay=weight_decay)
    elapsed_time('VB after training. starting Gamma_as_intervals')
    Yhat_val_vb_ =                          Gamma_as_intervals_1st_part(Y_seq_vb__val[:,:-m], [phi_vb_],m) # [N_val,d] -> [N_val, m, 1]
    Gamma_val_vb_ =                         Gamma_as_intervals_2nd_part(Yhat_val_vb_,lambda_vec,interval_common_ratio) # [N_val,m,1] -> [N_val,m,2,1,len_lambda]
    elapsed_time('VB after Gamma_as_intervals')
    Rhat_val_vb_ =                          torch.zeros_like(lambda_vec)
    for i_lambda in range(len_lambda):
        lambda_vb_ =                        lambda_vec[i_lambda]
        Gamma_lambda_val_vb_ =              Gamma_val_vb_[:,:,:,0,i_lambda] # k=0, representing the single fold. [N_val,m,2,1,len_lambda] -> [N_val,m,2]
        risk_val_vb_ =                      risk_ell(Y_seq_vb__val[:,-m:],Gamma_lambda_val_vb_)
        Rhat_val_vb_[i_lambda] =            1/(N_val+1) * (risk_val_vb_.sum() + B)
        if Rhat_val_vb_[i_lambda]<=alpha:
            elapsed_time(f'VB i_lambda={i_lambda}/{len_lambda} breaking the loop')
            break # argmin lambda s.t. (Rhat <= alpha)
    elapsed_time(f'   VB validation done, now testing Nte={N_te} sequences')
    assert(                                 lambda_vb_ < lambda_vec[-1] ) # make sure lambda was large enough for Rhat to drop below alpha
    Gamma_lambda_te__vb_ =                  Gamma_as_intervals(Y_seq_te[:,:-m], [phi_vb_], lambda_vb_, m, interval_common_ratio)
    Gamma_lambda_te__vb_ =                  dim_wise_union(Gamma_lambda_te__vb_) # not really needed for VB, here for uniformity with the CV methods
    risk_vb_ =                              risk_ell(Y_seq_te[:,-m:], Gamma_lambda_te__vb_)                  # output size = [ N_te              ]
    inef_vb_ =                              calc_ineff_of_intervals(Gamma_lambda_te__vb_)                    # output size = [ N_te, m           ]

    elapsed_time(f'VB             risk={risk_vb_.mean()} ineff={inef_vb_.mean()} lambda={lambda_vb_.mean()}')

    ## K-CV-CRC
    K =                                     args.num_folds
    elapsed_time(f'K-CV with K={K} folds')
    if (N_av % K == 0):                     # has meaning only when N/K is integer
        NoverK =                            int(N_av/K)
        phis_kcv =                          [deepcopy(init_net) for _ in range(K)]  # K different model parameters
        Yhat_val_kcv =                      torch.zeros(NoverK,m,K)
        for k in range(K):
            i_lfo =                         torch.cat((torch.arange(0,k*NoverK),torch.arange((k+1)*NoverK,N_av)))
            i_val =                         torch.arange(k * NoverK, (k + 1) * NoverK)  # the fold that is left out
            elapsed_time(f'    k={k}/{K} K-CV before training model')
            train_net(                      phis_kcv[k], Y_seq_av[i_lfo,:], lr=lr, num_iters=num_iters, weight_decay=weight_decay)
            Yhat_val_kcv[:,:,k] =           Gamma_as_intervals_1st_part(Y_seq_av[i_val,:-m], [phis_kcv[k]], m).view(NoverK, m)    # view [NoverK, m, 1  ] -> [NoverK, m]
        Gamma_val_kcv =                     Gamma_as_intervals_2nd_part(Yhat_val_kcv,lambda_vec, interval_common_ratio)                # size = [NoverK,m,K] -> [ NoverK, m, 2, K , len_lambda]
        Rhat_val_kcv =                      torch.zeros_like(lambda_vec)
        elapsed_time(f'    Looping lambda for {len_lambda} times')
        for i_lambda in range(len_lambda):
            lambda_kcv =                    lambda_vec[i_lambda]
            Rhat_val_kcv[i_lambda] =        B / (K+1)
            for k in range(K):
                i_val =                     torch.arange(k*NoverK,(k+1)*NoverK)  # the fold that is left out
                Gamma_lambda_val_kcv =      Gamma_val_kcv[:, :, :, k, i_lambda]  # size = [ NoverK, m, 2, K, len_lambda ]  -> [ NoverK, m, 2 ]
                Rhat_val_kcv[i_lambda] +=   ( risk_ell( Y_seq_av[i_val,-m:], Gamma_lambda_val_kcv ) ).mean(dim=0) /(K+1) # mean over NoverK
            if (i_lambda%10==1):
                elapsed_time(f'    i_lambda={i_lambda}/{len_lambda} Rhat_val_kcv[i_lambda]={Rhat_val_kcv[i_lambda]} ; alpha={alpha}')
            if Rhat_val_kcv[i_lambda]<=alpha:
                elapsed_time(f'    i_lambda={i_lambda}/{len_lambda} breaking the loop Rhat_val_kcv[i_lambda]={Rhat_val_kcv[i_lambda]} <= {alpha}=alpha')
                break # argmin lambda s.t. (Rhat <= alpha)
        elapsed_time(f'   K-CV validation done, now testing Nte={N_te} sequences')
        assert(                             lambda_kcv < lambda_vec[-1] ) # make sure lambda was large enough for Rhat to drop below alpha
        Gamma_lambda_te__kcv =              Gamma_as_intervals(Y_seq_te[:,:-m], phis_kcv, lambda_kcv, m, interval_common_ratio)    # output size = [ N_te, m, 2,    K  ]
        Gamma_lambda_te__kcv =              dim_wise_union(Gamma_lambda_te__kcv)                            # output size = [ N_te, m, 2*K      ]
        risk_kcv =                          risk_ell(Y_seq_te[:,-m:], Gamma_lambda_te__kcv)                 # output size = [ N_te              ]
        inef_kcv =                          calc_ineff_of_intervals(Gamma_lambda_te__kcv)                   # output size = [ N_te, m           ]
    else:
        raise Exception((                   'N/K must be integer'))

    elapsed_time(f'VB             risk={risk_vb_.mean()} ineff={inef_vb_.mean()} lambda={lambda_vb_.mean()}')
    elapsed_time(f'K-CV (K={K   })  risk={risk_kcv.mean()} ineff={inef_kcv.mean()} lambda={lambda_kcv.mean()}')

    ## N-CV-CRC
    K =                                     N_av
    elapsed_time(f'N-CV with K={N_av} folds')
    if (N_av % K == 0):                     # has meaning only when N/K is integer, this is always true, but is kept to be of similar code to K-CV-CRC with K=N as a special case
        NoverK =                            int(N_av/K)
        phis_ncv =                          [deepcopy(init_net) for _ in range(K)]  # K different model parameters
        Yhat_val_ncv =                      torch.zeros(NoverK,m,K)
        for k in range(K):
            i_lfo =                         torch.cat((torch.arange(0,k*NoverK),torch.arange((k+1)*NoverK,N_av)))
            i_val =                         torch.arange(k * NoverK, (k + 1) * NoverK)  # the fold that is left out
            elapsed_time(f'    n={k}/{K} N-CV before training model')
            train_net(                      phis_ncv[k], Y_seq_av[i_lfo,:], lr=lr, num_iters=num_iters, weight_decay=weight_decay)
            Yhat_val_ncv[:,:,k] =           Gamma_as_intervals_1st_part(Y_seq_av[i_val,:-m], [phis_ncv[k]], m).view(NoverK, m)    # view [NoverK, m, 1  ] -> [NoverK, m]
        Gamma_val_ncv =                     Gamma_as_intervals_2nd_part(Yhat_val_ncv,lambda_vec,interval_common_ratio)                # size = [NoverK,m,K] -> [ NoverK, m, 2, K , len_lambda]
        Rhat_val_ncv =                      torch.zeros_like(lambda_vec)
        elapsed_time(f'    Looping lambda for {len_lambda} times')
        for i_lambda in range(len_lambda):
            lambda_ncv =                    lambda_vec[i_lambda]
            Rhat_val_ncv[i_lambda] =        B / (K+1)
            for k in range(K):
                i_val =                     torch.arange(k*NoverK,(k+1)*NoverK)  # the fold that is left out
                Gamma_lambda_val_ncv =      Gamma_val_ncv[:, :, :, k, i_lambda]  # size = [ NoverK, m, 2, K, len_lambda ]  -> [ NoverK, m, 2 ]
                Rhat_val_ncv[i_lambda] +=   ( risk_ell( Y_seq_av[i_val,-m:], Gamma_lambda_val_ncv ) ).mean(dim=0) /(K+1) # mean over NoverK
            if (i_lambda%10==1):
                elapsed_time(f'    i_lambda={i_lambda}/{len_lambda} Rhat_val_ncv[i_lambda]={Rhat_val_ncv[i_lambda]} ; alpha={alpha}')
            if Rhat_val_ncv[i_lambda]<=alpha:
                elapsed_time(f'    i_lambda={i_lambda}/{len_lambda} breaking the loop Rhat_val_ncv[i_lambda]={Rhat_val_ncv[i_lambda]} <= {alpha}=alpha')
                break # argmin lambda s.t. (Rhat <= alpha)
        elapsed_time(f'   N-CV validation done, now testing Nte={N_te} sequences')
        assert(                             lambda_ncv < lambda_vec[-1] ) # make sure lambda was large enough for Rhat to drop below alpha
        Gamma_lambda_te__ncv =              Gamma_as_intervals(Y_seq_te[:,:-m], phis_ncv, lambda_ncv, m,interval_common_ratio)    # output size = [ N_te, m, 2,    K  ]
        Gamma_lambda_te__ncv =              dim_wise_union(Gamma_lambda_te__ncv)                            # output size = [ N_te, m, 2*K      ]
        risk_ncv =                          risk_ell(Y_seq_te[:,-m:], Gamma_lambda_te__ncv)                 # output size = [ N_te              ]
        inef_ncv =                          calc_ineff_of_intervals(Gamma_lambda_te__ncv)                   # output size = [ N_te, m           ]
    else:
        raise Exception((                   'N/K must be integer'))


    elapsed_time(f'VB           risk={risk_vb_.mean()} ineff={inef_vb_.mean()} lambda={lambda_vb_.mean()}')
    elapsed_time(f'K-CV (K={args.num_folds})  risk={risk_kcv.mean()} ineff={inef_kcv.mean()} lambda={lambda_kcv.mean()}')
    elapsed_time(f'N-CV (N={N_av})  risk={risk_ncv.mean()} ineff={inef_ncv.mean()} lambda={lambda_ncv.mean()}')

    file_name =                             f'output_N_{N_av}_seed_{seed}.mat'
    elapsed_time(f'saving to file {file_name}')
    savemat(file_name=  file_name,
            mdict=      {   'seed'                  : seed,
                            'N_av'                  : N_av,
                            'N_te'                  : N_te,
                            'alpha'                 : alpha,
                            'lr'                    : lr,
                            'lambda_vec'            : lambda_vec.numpy(),
                            'm'                     : m,
                            'K'                     : args.num_folds,
                            'Y_seq_te'              : Y_seq_te.numpy(),
                            'Gamma_lambda_te__vb_'  : Gamma_lambda_te__vb_.numpy(),
                            'Rhat_val_vb_'          : Rhat_val_vb_.numpy(),
                            'risk_vb_'              : risk_vb_.numpy(),
                            'inef_vb_'              : inef_vb_.numpy(),
                            'lambda_vb_'            : lambda_vb_.numpy(),
                            'Gamma_lambda_te__kcv'  : Gamma_lambda_te__kcv.numpy(),
                            'Rhat_val_kcv'          : Rhat_val_kcv.numpy(),
                            'risk_kcv'              : risk_kcv.numpy(),
                            'inef_kcv'              : inef_kcv.numpy(),
                            'lambda_kcv'            : lambda_kcv.numpy(),
                            'Gamma_lambda_te__ncv'  : Gamma_lambda_te__ncv.numpy(),
                            'Rhat_val_ncv'          : Rhat_val_ncv.numpy(),
                            'risk_ncv'              : risk_ncv.numpy(),
                            'inef_ncv'              : inef_ncv.numpy(),
                            'lambda_ncv'            : lambda_ncv.numpy(),
                        })

    elapsed_time('end of main')
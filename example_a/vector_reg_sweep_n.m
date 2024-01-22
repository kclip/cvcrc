%%
clear all; clc; close all;

%%

K =                     20;             % K-fold for cross-validation
v_N =                   [40:40:520];    % Data set size
num_N =                 length(v_N);
num_sim =               50;             % number of data set drawn, each called a run
d =                     50;             % dim of x
m =                     30;             % dim of y
N_te =                  200;            % 100 test inputs per run
alpha =                 0.1;            % miscoverage level
mu_0 =                  10;
beta_0 =                4;              % ground truth
gamma_0 =               1;
B =                     1;              % upper bound of the risk, which is the within-cube-ratio 
lambda_vec =            0.1:0.1:200;
len_lambda =            length(lambda_vec);

m_malloc =              nan(num_N ,num_sim, N_te);
s_malloc =              struct('risk',m_malloc,'inef',m_malloc);
% In the following, s for struct; risk = risk; inef = inefficiency;
s_vb_ =                 s_malloc; % vb_ = validation based (split)
s_ncv =                 s_malloc; % ncv = N-fold cross-validation (jackknife+)
s_kcv =                s_malloc; % kcv = K-fold cross-validation (K-fold jackknife+)

N_last =                v_N(num_N);
set(0,'defaultTextInterpreter','latex','defaultAxesFontSize',14,'defaultLegendFontSize',14);

for i_s = 1:num_sim
    phi_groundtruth =                       mu_0       * sign(rand(d,m)-0.5) + 1/sqrt(gamma_0) * randn(d,       m); % ground-truth
    X_complete =                            0                                + 1/sqrt(d)       * randn(N_last,  d);
    X_te  =                                 0                                + 1/sqrt(d)       * randn(N_te,    d);
    Y_complete =                            X_complete * phi_groundtruth     + 1/sqrt(beta_0)  * randn(N_last,  m);  % ground-truth
    Y_te =                                  X_te       * phi_groundtruth     + 1/sqrt(beta_0)  * randn(N_te,    m);  % ground-truth
    for i_N=1:num_N
        disp(['i_s=',num2str(i_s),'/',num2str(num_sim),' i_N=',num2str(i_N),'/',num2str(num_N)]);
        N =                                 v_N(i_N);
        X =                                 X_complete(1:N, 1:d); % ground-truth
        Y =                                 Y_complete(1:N, 1:m); % ground-truth
        %% vb_ = validation based (split)
        N_tr =                              floor(N*0.5);%max(d,floor(N/2));                      % num of proper training points
        N_val =                             N - N_tr;
        assert(                             N_val >= 1/alpha );
        X_tr  =                             X( 1        : N_tr , 1:d );             % tr =  proper training
        X_val =                             X( N_tr + 1 : end  , 1:d );             % val = validation
        Y_tr  =                             Y( 1        : N_tr , 1:m );
        Y_val =                             Y( N_tr + 1 : end  , 1:m );
        phi_vb__ml_ =                       fitting_ls(     X_tr ,   Y_tr );                    % model phi; data set = training; alg = max likelihood ML
        Rhat_val_vb_ =                      nan(len_lambda,1);
        for i_lambda=1:len_lambda
            lambda_vb_ =                    lambda_vec(i_lambda);
            Gamma_lambda_val_vb_ =          Gamma_as_intervals(X_val, phi_vb__ml_, lambda_vb_);
            risk_val_ml_ =                  risk_ell(Y_val,Gamma_lambda_val_vb_);
            Rhat_val_vb_(i_lambda) =        1/(N_val+1) * (sum(risk_val_ml_) + B);
            if Rhat_val_vb_(i_lambda)<=alpha
                break; % argmin lambda s.t. (Rhat <= alpha)
            end
        end
        assert(                             lambda_vb_ < lambda_vec(end) ); % make sure lambda was large enough for Rhat to dro below alpha
        Gamma_lambda_te__vb_ =              Gamma_as_intervals(X_te, phi_vb__ml_, lambda_vb_);
        Gamma_lambda_te__vb_ =              dim_wise_union(Gamma_lambda_te__vb_); % not really needed for VB, here for uniformity with the CV methods
        s_vb_.risk(i_N,i_s,:) =             risk_ell(Y_te, Gamma_lambda_te__vb_);
        s_vb_.inef(i_N,i_s,:) =             calc_ineff_of_intervals(Gamma_lambda_te__vb_);
        
        %% ncv = jackknife+
        m_phi_ncv_ml_ =                     nan(d,     m,  N);
        for i = 1:N                                                     % each sample is a fold of its own
            i_loo =                         [1:i-1,i+1:N];              % loo = leave-one-out
            m_phi_ncv_ml_(1:d,1:m,i) =      fitting_ls( X(i_loo, 1:d ),...      % tr =  proper training
                                                        Y(i_loo, 1:m ));        % model phi; data set = training; alg = max likelihood ML
        end
        Rhat_val_ncv =                      nan(len_lambda,1);
        for i_lambda=1:len_lambda
            lambda_ncv =                    lambda_vec(i_lambda);
            Rhat_val_ncv(i_lambda) =        B /(N+1);
            for i = 1:N
                Gamma_lambda_val_ncv =      Gamma_as_intervals(X(i, 1:d ), m_phi_ncv_ml_(1:d,1:m,i), lambda_ncv);
                Rhat_val_ncv(i_lambda) =    Rhat_val_ncv(i_lambda) + risk_ell(Y( i, 1:m ), Gamma_lambda_val_ncv ) /(N+1);
            end
            if Rhat_val_ncv(i_lambda)<=alpha
                break; % argmin lambda s.t. (Rhat <= alpha)
            end
        end
        assert(                             lambda_ncv < lambda_vec(end) ); % make sure lambda was large enough for Rhat to dro below alpha
        Gamma_lambda_te__ncv =              Gamma_as_intervals(X_te, m_phi_ncv_ml_, lambda_ncv);
        Gamma_lambda_te__ncv =              dim_wise_union(Gamma_lambda_te__ncv);
        s_ncv.risk(i_N,i_s,:) =             risk_ell(Y_te, Gamma_lambda_te__ncv);
        s_ncv.inef(i_N,i_s,:) =             calc_ineff_of_intervals(Gamma_lambda_te__ncv);

        %% kcv = K-fold jackknife+
        if rem(N,K)==0                      % has meaning only when N/K is integer
            NoverK =                        round(N/K);
            m_phi_kcv_ml_ =                nan(d,       m,  K); % lfo = leave-fold-out (N/K samples out)
            for k = 1:K
                i_lfo =                     [1 : (k-1)*NoverK  ,  k*NoverK+1 : N];
                m_phi_kcv_ml_(1:d,1:m,k) = fitting_ls( X(i_lfo, 1:d ),...
                                                        Y(i_lfo, 1:m));
            end
            Rhat_val_kcv =                 nan(len_lambda,1);
            for i_lambda=1:len_lambda
                lambda_kcv =               lambda_vec(i_lambda);
                Rhat_val_kcv(i_lambda) =   B / (K+1);
                for k = 1:K
                    i_val =                 (k-1)*NoverK+ 1 : k*NoverK ;  % the fold that is left out
                    Gamma_lambda_val_kcv = Gamma_as_intervals(X(i_val, 1:d ), m_phi_kcv_ml_(1:d,1:m,k), lambda_kcv);
                    Rhat_val_kcv(i_lambda) = Rhat_val_kcv(i_lambda) + mean( risk_ell(Y( i_val, 1:m ), Gamma_lambda_val_kcv  ) ) /(K+1);
                end
                if Rhat_val_kcv(i_lambda)<=alpha
                    break; % argmin lambda s.t. (Rhat <= alpha)
                end
            end
            assert(                             lambda_kcv < lambda_vec(end) ); % make sure lambda was large enough for Rhat to dro below alpha
            Gamma_lambda_te__kcv =             Gamma_as_intervals(X_te, m_phi_kcv_ml_, lambda_kcv);
            Gamma_lambda_te__kcv =             dim_wise_union(Gamma_lambda_te__kcv);
            s_kcv.risk(i_N,i_s,:) =            risk_ell(Y_te, Gamma_lambda_te__kcv);
            s_kcv.inef(i_N,i_s,:) =            calc_ineff_of_intervals(Gamma_lambda_te__kcv);
        else
            error(                          'N/K must be integer');
        end

        if (mod(i_s,10)==1) && (i_N==5) % for plotting
            figure;
            set(gcf,'Position',[-1020,-120,1000,1200]);
            m_vis =                 m;%30;
            n_vis =                 1;
            %% %%%%%%%%%%%%%  VB-CRC
            subplot(1,3,1); title('VB-CRC'); 
            Yhat_vb__ml_ =          model_predict(X_te, phi_vb__ml_);
            for j=1:m_vis
                rectangle('Position', [Gamma_lambda_te__vb_(n_vis,j,1), j-0.4, Gamma_lambda_te__vb_(n_vis,j,2)-Gamma_lambda_te__vb_(n_vis,j,1), 0.8], 'FaceColor', 'c', 'EdgeColor', 'c'); hold on;
            end
            plot(Yhat_vb__ml_(n_vis,1:m_vis),1:m_vis,'mx','MarkerSize',6);
            plot(Y_te(        n_vis,1:m_vis),1:m_vis,'kx','MarkerSize',12);
            %%%%%%%%%%%%%%%   N-CV-CRC
            subplot(1,3,2); title(['$N=',num2str(N),'$-CV-CRC']);
            for j=1:m_vis
                for k=1:size(Gamma_lambda_te__ncv,3)/2
                    if ~isinf(Gamma_lambda_te__ncv(n_vis,j,2*k-1))
                        rectangle('Position', [Gamma_lambda_te__ncv(n_vis,j,2*k-1), j-0.4, Gamma_lambda_te__ncv(n_vis,j,2*k)-Gamma_lambda_te__ncv(n_vis,j,2*k-1), 0.8], 'FaceColor', 'c', 'EdgeColor', 'c'); hold on;
                    end
                end
            end
            for k=1:N
                Yhat_ncv_ml_ =          model_predict(X_te, m_phi_ncv_ml_(:,:,k));
                plot(Yhat_ncv_ml_(n_vis,1:m_vis),1:m_vis,'mx','MarkerSize',6);        
            end
            plot(Y_te(        n_vis,1:m_vis),1:m_vis,'kx','MarkerSize',12);
            %%%%%%%%%%%%%%%   K-CV-CRC
            subplot(1,3,3); title(['$K=',num2str(K),'$-CV-CRC']);
            for j=1:m_vis
                for k=1:size(Gamma_lambda_te__kcv,3)/2
                    if ~isinf(Gamma_lambda_te__kcv(n_vis,j,2*k-1))
                        rectangle('Position', [Gamma_lambda_te__kcv(n_vis,j,2*k-1), j-0.4, Gamma_lambda_te__kcv(n_vis,j,2*k)-Gamma_lambda_te__kcv(n_vis,j,2*k-1), 0.8], 'FaceColor', 'c', 'EdgeColor', 'c'); hold on;
                    end
                end
            end
            for k=1:size(Gamma_lambda_te__kcv,3)/2
                Yhat_kcv_ml_ =         model_predict(X_te, m_phi_kcv_ml_(:,:,k));
                plot(Yhat_kcv_ml_(n_vis,1:m_vis),1:m_vis,'mx','MarkerSize',6);        
            end
            plot(Y_te(n_vis,1:m_vis),1:m_vis,'kx','MarkerSize',12);
        end
    end % i_N
        
    %%

    v_N_kcv =                           (mod(v_N,K)==0);
    s_vb_.inef(isinf(s_vb_.inef)) =     200; % for plotting

    [risk_vb__avg,  risk_vb__std] =     mean_std_analysis(  s_vb_.risk, i_s);
    [risk_ncv_avg,  risk_ncv_std] =     mean_std_analysis(  s_ncv.risk, i_s);
    [risk_kcv_avg,  risk_kcv_std] =     mean_std_analysis(  s_kcv.risk(v_N_kcv,:,:,:), i_s); % consider only vaild indices for kcv

    [inef_vb__avg,  inef_vb__std] =     mean_std_analysis(  s_vb_.inef, i_s);
    [inef_ncv_avg,  inef_ncv_std] =     mean_std_analysis(  s_ncv.inef, i_s);
    [inef_kcv_avg,  inef_kcv_std] =     mean_std_analysis(  s_kcv.inef(v_N_kcv,:,:,:), i_s); % consider only vaild indices for kcv

    save ws_toy_Gaussians_sweep_n;
    if length(v_N)>1
        figure(1001); cla reset; set(gca,'XScale','log');
        plot_curves_with_conf_intervals(v_N,           risk_vb__avg  , risk_vb__std  , 'b', 'o-' , 'VB' );
        plot_curves_with_conf_intervals(v_N(v_N_kcv), risk_kcv_avg , risk_kcv_std , 'c', 'd-' , ['$K$-CV ($K=',num2str(K),'$)'] );
        plot_curves_with_conf_intervals(v_N,           risk_ncv_avg  , risk_ncv_std  , 'r', '^--', '$N$-CV ($K=N$)');
    
        plot(v_N, alpha*ones(size(v_N)) ,'g--','DisplayName','\alpha');
        xlabel('Data set size $N$'); ylabel('Risk'); legend show; grid;
        set(gca,'XLim',[v_N(1),v_N(end)]);
        set(gca,'YLim',[0,1.2*alpha]);
        set(gcf,'Position',[400,50,560,420]);
    
        figure(1002); cla reset;
        plot_curves_with_conf_intervals(v_N,           inef_vb__avg  , inef_vb__std  , 'b' , 'o-' , 'VB'  );
        plot_curves_with_conf_intervals(v_N(v_N_kcv), inef_kcv_avg , inef_kcv_std , 'c' , 'd-' , ['KCV (K=',num2str(K),')'] );
        plot_curves_with_conf_intervals(v_N,           inef_ncv_avg  , inef_ncv_std  , 'r' , '^--', 'NCV (K=N)' );
        
        xlabel('Data set size $N$'); ylabel('Inefficiency'); legend show; grid;
        set(gca,'YScale','log'); yticks([2 4 8 10 20 40 50]);
        set(gca,'XLim',[v_N(1),v_N(end)]);
        set(gca,'YLim',[1.8,50]);
        set(gcf,'Position',[960,50,560,420]);
    end

end % i_s


function phi = fitting_ls(X,Y)
[N,d] =             size(X);
reg_value =         0.1;
if N>=d% more equations than variables
    phi =           (X.' *   X + reg_value * eye(d) ) \ (X.'   * Y); % LS solution
else % less equations than variables, choosing the one with the min norm
    phi =           X.'  *(( X  *  X.'  + reg_value * eye(N) ) \ Y); % LS solution with min norm
end
end

function out = model_predict(X,phi) % X is [N,d], phi is [d,m]
out =               X * phi;
end

function [m_avg, m_std] = mean_std_analysis(m_in, i_s)
m_avg =             squeeze(mean( m_in(:,1:i_s,:),  [2,3]));
m_std =             squeeze(std(  m_in(:,1:i_s,:),1,[2,3])/sqrt(i_s));
end


function Gamma = Gamma_as_intervals(X, phi, lambda)
% size(X)       [ N, d        ]
% size(phi)     [ d, m    , K ]
% size(lambda)  [ 1, 1        ]
% size(Gamma)   [ N, m, 2, K  ]  Only one interval
[N,d] =                     size(X);
K =                         size(phi, 3);
m =                         size(phi, 2);
Gamma =                     nan(N, m, 2, K);
for k=1:K
    Yhat =                  model_predict(X, phi(1:d,1:m,k));
    Gamma(1:N,1:m,1,k) =    Yhat - lambda/2; 
    Gamma(1:N,1:m,2,k) =    Yhat + lambda/2;
end

end

function risk = risk_ell(Y, Gamma)
% size(Y)       [ N , m      ]
% size(Gamma)   [ N , m, 2*K ]
% size(risk)    [ N , 1      ]
[N,m] =                     size(Y);
K =                         size(Gamma, 3)/2;
is_y_in_Gamma =             zeros(N,m,K);
for n=1:N
    for j=1:m
        for k=1:K
            is_y_in_Gamma(n,j,k) =  ( (Y(n,j) >= Gamma(n,j,2*k-1)) & (Y(n,j) <= Gamma(n,j,2*k)) ); % element-wise logical and
        end
    end
end
risk =      mean(~any(is_y_in_Gamma,3),2);
end

function GammaOut = dim_wise_union(GammaIn)
% size(GammaIn)    [ N , m, 2  , K ]
% size(GammaOut)   [ N , m, 2*K    ] # null intervals will be located at the end with [inf,inf] represenation
[N,m,~,K] =                 size(GammaIn);
GammaOut =                  inf(N,m,2*K);
for n=1:N
    for j=1:m
        borders=            unique(sort(reshape(GammaIn(n,j,:,:),1,[])));
        representors =      0.5*([-inf,borders]+[borders,inf]);
        mask =              zeros(size(representors));
        for k=1:K
            mask =          mask | (  (representors >= GammaIn(n,j,1,k)) & (representors <= GammaIn(n,j,2,k) )  );
        end
        diff_mask =         diff(mask);
        risings =           borders(diff_mask==1);
        %if length(risings)>1
        %    disp(num2str(risings));
        %end
        fallings =          borders(diff_mask==-1); % length should be the same as risings
        GammaOut(n,j,2*(1:length(risings ))-1) =    risings;
        GammaOut(n,j,2*(1:length(fallings))  ) =    fallings;
    end
end
end

function ineff = calc_ineff_of_intervals(Gamma) % mean over samples of the log of the d-dim volume
% size(Gamma)       [ N , m, 2*K    ]
ineff =                     mean(sum(Gamma(:,:,2:2:end)-Gamma(:,:,1:2:end),[3],'omitnan'),[1,2]); % omit NaN for the unused intervals which are inf-inf
end
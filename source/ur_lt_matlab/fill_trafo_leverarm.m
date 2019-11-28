function [adj] = fill_trafo_leverarm(adj,var)
%FILL_TRAFO_LEVERARM fills the functional model defined in
% fct_trafo_leverarm.m with the observations, approximate values and the
% variances
%
%   Input:
%      adj     ... structure including various adjustment parameters
%      var     ... defines the stochastic model (var.cov) as well if
%                  identical points will be considered (var.identPts)
%   Output:
%      adj.A
%      adj.B
%      adj.w
%      adj.Qll ... are filled and build up

% Sabine Horvath, TU Wien, 22.05.2019


adj.B = zeros(adj.bed,adj.beob); % Reset
x = adj.x;
ll = adj.ll;

if ~var.leverarmRF
    j = 1; k = 1;
    kk = adj.samples*3*2+1;
    for i = 1:adj.samples
        
        A1 = adj.aa(x(1),x(2),x(3),ll(k),ll(k+1),ll(k+2),ll(k+3),ll(k+4),...
            ll(k+5),x(7),x(8),x(9)); %[x(1:3);ll(k:k+5);x(7:9)]'
        
        B1 = adj.bb(x(1),x(2),x(3),ll(k),ll(k+1),ll(k+2),x(7),x(8),x(9));
        
        w1 = adj.gg(x(1),x(2),x(3),ll(k),ll(k+1),ll(k+2),ll(k+3),ll(k+4),...
            ll(k+5),ll(k+6),ll(k+7),ll(k+8),x(4),x(5),x(6),x(7),...
            x(8),x(9));
        
        % Covariances of Obs
        if var.cov == 1
            sll(1:3,1:3) = adj.Cll_eul;
            %             qll(4:6,4:6) = diag(Kov(j:j+2,:))'.*eye(3,3); %Qll_LT; *20 %
            sll(4:6,4:6) = adj.Kov(j:j+2,:); %*400; %Qll_LT; % 400
            sll(7:9,7:9) = adj.Cll_R;
            adj.Qll(k:k+9-1,k:k+9-1) = sll*1/adj.s02_apri;
        end
        
        
        adj.A(j:j+2,1:9) = A1; %eval(A1);
        adj.w(j:j+2,1) = w1; %eval(w1);
        adj.B(j:j+2,k:k+9-1) = B1; %eval(B1);
        k = k+9;
        
        j = j+3;
    end
    
    
elseif var.leverarmRF
    j = 1; k = 1;
    kk = adj.samples*3*2+1;
    for i = 1:adj.samples
        
        A1 = adj.aa(x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),ll(k+3),...
            ll(k+4),ll(k+5),ll(k+6),ll(k+7),ll(k+8));
        
        B1 = adj.bb(x(1),x(2),x(3),x(7),x(8),x(9),ll(k+3),ll(k+4),ll(k+5));
        
        w1 = adj.gg(x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),ll(k),ll(k+1),ll(k+2),ll(k+3),ll(k+4),ll(k+5),ll(k+6),ll(k+7),ll(k+8));
        
        
        % Covariances of Obs
        if var.cov == 1
            sll(1:3,1:3) = adj.Kov(j:j+2,:); %.*eye(3,3);
            sll(4:6,4:6) = adj.Cll_R_axang; %*400; %Qll_LT; % 400
            sll(7:9,7:9) = adj.Cll_R;
            adj.Qll(k:k+9-1,k:k+9-1) = sll*1/adj.s02_apri;
        end
        
        
        adj.A(j:j+2,1:9) = A1; %eval(A1);
        adj.w(j:j+2,1) = w1; %eval(w1);
        adj.B(j:j+2,k:k+9-1) = B1; %eval(B1);
        k = k+9;
        
        j = j+3;
    end

    
end


end


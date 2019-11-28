function [adj] = gauss_helmert(adj)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


% Solve GH
adj.ww_it = adj.ww;
adj.ww = adj.w+adj.B*(adj.l0-adj.ll); %Second term: see Pope (1972)


adj.Nkk = pinv(adj.B*adj.Qll*adj.B'); % corresp. to Q11!
adj.xa = -(adj.A'*adj.Nkk*adj.A)\(adj.A'*adj.Nkk*adj.ww);
adj.va = adj.Qll*adj.B'*(adj.Nkk*adj.A*pinv(adj.A'*adj.Nkk*adj.A)*adj.A'*adj.Nkk-adj.Nkk)*adj.ww;

adj.x1 = adj.x;
adj.x = adj.x + adj.xa;
adj.ll = adj.l0+adj.va;


% Stochastics
adj.frei = adj.bed-adj.u;
adj.s02_apost = adj.va'*pinv(adj.Qll)*adj.va./adj.frei;

Qxx = pinv(adj.A'*adj.Nkk*adj.A); %%% adapting!!! -> Nkk
adj.Cxx = Qxx*adj.s02_apost; %%% adapting!!! -> Nkk
adj.stdX = sqrt(diag(adj.Cxx));

Qkka = adj.Nkk - adj.Nkk*adj.A*Qxx*adj.A'*adj.Nkk;
adj.Cll = (adj.Qll - adj.Qll*adj.B'*Qkka*adj.B*adj.Qll)*adj.s02_apost; % Qll - Qvv
adj.stdL = sqrt(diag(adj.Cll));

adj.Cvv = (adj.Qll*adj.B'*Qkka*adj.B*adj.Qll)*adj.s02_apost;
adj.stdv = sqrt(diag(adj.Cvv));


% Global test
alpha = 0.05;
testval = adj.s02_apost/adj.s02_apri;
fqu = [finv(alpha/2,adj.frei,inf) finv(1-alpha/2,adj.frei,inf)];
if testval > fqu(1) && testval < fqu(2)
    adj.gt = 1;
else
    adj.gt = 0;
end


end


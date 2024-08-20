function [A,S,W,H] = vemove(X,W,H,Kj,maxNumIter)
%VEMOVE    Variational EM for separation of mixtures with moving sources.
%        
% INPUTS
%
%  X   : [F x L x I]         F <freq> x L <frams> complex STFT of I sensors
%
%  W   : [F x K]             initial matrix of bases (of all K components)
%
%  H   : [K x L]             initial matrix of contributions (all K comp.)
%
%  Kj  : {J x 1} x [? x 1]   Kj is a cell-array with J elements (one for 
%                            each source). Let K = size(W,2) be the total
%                            number of components. Any element Kj{j}, j=1:J
%                            of Kj contains the indexes (of the columns of
%                            W and rows of H) that correspond to source j.
%                            For example Kj = { [1 2 3] , [4 5] , [6 7] }
%                            would set the number of sources to J = 3,
%                            and use W(:,[1 2 3]) * H([1 2 3],:) as the NMF
%                            for souce j=1, W(:,[4 5]) * H([4 5],:) as
%                            the NMF of source j=2, W(:,[6 7]) * H(:,[6 7])
%                            as the NMF for source j=3.
%
%  maxNumIter : [1 x 1]      number of EM iterations
%
% OUTPUTS
%
%  A  : [F x L x I x J]      estimated mixing matrices (up to scale)
%
%  S  : [F x L x J]          estimated monochannel sources (up to scale)
% 
%  W  : [F x K]              estimated matrix of bases
%
%  H  : [K x L]              estimated matrix of contributions
%
%  References:
%    [1] D. Kounades-Bastian, L. Girin, X. Alameda-Pineda, S. Gannot,
%        R. Horaud, A Variational EM Algorithm for the Separation of Time-
%        Varying Convolutive Audio Mixtures, IEEE/ACM TASLP, 2016.
%
%    [2] D. Kounades-Bastian, L. Girin, X. Alameda-Pineda, S. Gannot, 
%        R. Horaud, A Variational EM Algorithm for the Separation of Moving
%        Sound Sources, WASPAA, 2015.
%
% version April 2, 2016 9:26 PM
fprintf('[vemove] v. April 2, 2016 9:26 PM\n');




%    /\
%   /__\
%  /    \ R C H E T Y P E



%% A   constants indexSets & functions

[F,L,I] = size(X);  J = numel(Kj);  IJ = I*J;   b1 = 1:IJ; b2 = IJ+1:2*IJ;

% [K x 1] inverse-search (partition of components to sources)
jk = zeros(size(W,2),1);    for j=1:J, jk( Kj{j} ) = j; end

% [J x J] linear indices from 1 : J^2
D = reshape( 1:J*J ,J,J );

% {J x J} x [I x 1] linear indices to extract the diagonals of blks in Qa
%
% CONSTRUCTION_____________________________________________________________
%   if Qa is [IJ x IJ] then Y = Qa(blk) is an [J x J x I] array whose
%   fiber Y(r,j,:) contains the diagonal elements of the {jr}^th [I x I]
%   blk of Qa. Thenafter sum(Qa(blk),3) yields the [J x J] matrix U 
%   already transposed NOTE THAT U_rj = tr{Q_jr} THE TRANSPOSED BLOCK OF Qa
%__________________________________________________________________________
blk = arrayfun(@(blk) find( kron( transpose(D) , eye(I) ) == blk ), D, 'uniformoutput', false);

% [J x J x I] cast into array
blk = permute( reshape( cat(1,blk{:}) , [I J J] ) , [2 3 1] );

% [IJ x IJ] alternative construction for kron() with a diagonal matrix
%
% CONSTRUCTION_____________________________________________________________
%   If Qs is [J x J] then  Qs(cloneTranspose).*onesKronIden   is equivalent
%   with      kron( transpose(Qs) , eye(I) ) .   In detail cloneTranpose is
%   an [IJ x IJ] array of lin-inds from 1:J^2, each one is replicated in a
%   [IxI] blk in TRANSPOSE ORDER, onesKronIden eliminates non diag elements
%__________________________________________________________________________
cloneTranspose = kron( transpose(D) , ones(I) );

% [IJ x IJ] see above
onesKronIden = repmat( eye(I) , J );

% f(x) symmetricity may be lost with the course of iterations
symtricize = @(A) .5 * (A+A');

% f(x) convert an {F x L} x [J x 1] cell in an [F x L x J] numerical array
%
% CONSTRUCTION_____________________________________________________________
%   In general the input cell A can be {F x L} x [IJ x I], where I>=1.
%   Technically if A : {F x L} x [IJ x 1] then B = cell2array(A) results in
%   an [F x L x J x I] array B where for A,B holds
%   reshape(A{f,l},I,J) = transpose(squeeze(B(f,l,:,:))).
%__________________________________________________________________________
cell2array = @(A) permute( reshape(cat(1,A{:}) ,[],J,F,L ) , [3 4 2 1] );
%%




%   _____
%     |
%     |
%   __|__  I N I T I A L I S A T I O N



%% I   Initialisation & offsets

% INITIALISE
%   v       : {F x L} x [ 1 x 1 ] sensor noise
%   Va      : {F x L} x [IJ x IJ] posterior covariance of vec(A)
%   eV      : {F}     x [IJ x IJ] evolution covariance of vec(A)
%   u       : [F x L x K]         prior variance of C

% {F x L} x [I x I] sensor noise
v = cell(F,L);    v(:)  = { X(:)'*X(:) / numel(X) };

% {F x L} x [IJ x 1] posterior mean of vec(A)
A = cell(F,L);    A(:)  = { ones(IJ,1) };

% {F} x [IJ x IJ] evolution covariance
eV = cell(F,1);   eV(:) = { eye(IJ) };

% {F x L} x [IJ x IJ] channel's posterior covariance
Va = cell(F,L);   Va(:) = { eye(IJ) };

% [1 x L x K] matrix of contributions (permute in convenient dimensions)
H = permute( H , [3 2 1] );

% [F x L x K] prior variance of C
u = bsxfun(@times, permute(W,[1 3 2]), H );

% {F x L} x [I x 1] convert X in cell for convenient multiplication
X = permute( num2cell( permute(X,[3 1 2]) , 1 ) , [2 3 1]);
%%







for iter = 1:maxNumIter
%   ____
%  |
%  |____
%  |
%  |____ - S    S T E P



%% E-S   source inference

% UPDATE
%   U   : {F x L} x [J x J]   posterior expectation of (A^H * A)
%   b   : [F x L  x J]        inverse of sum of u over Kj
%   Vs  : {F x L} x [J x J]   source posterior covariance
%   S   : {F x L} x [J x 1]   source estimate

% {F x L} x [IJ x IJ] here U is E[vec(A)*vec(A)']
U = cellfun(@(Va,A) Va+A*A', Va,A, 'uniformoutput', false);
    
% {F x L} x [J x J] U is E[A^H*A] by contraction of itself
U = cellfun(@(U) symtricize( sum( U(blk) , 3 ) ), U, 'uniformoutput', false);

% {J x 1} x [F x L] prior psd of S
b = cellfun(@(Kj) sum(u(:,:,Kj),3), Kj, 'uniformoutput', false);

% [F x L x J] prior precision of S
b = 1 ./ cat(3,b{:});

% {F x L} x [J x J] posterior covariance of S
Vs = cellfun(@(b,U,v) inv( diag(b(:)) + U/v ), num2cell(b,3), U, v, 'uniformoutput', false);

% {F x L} x [J x 1] source estimates
S = cellfun(@(Vs,A,X,v) Vs * ( reshape(A,I,J)' * X/v ), Vs,A,X,v, 'uniformoutput', false);
%%




%  |\  /|
%  | \/ | - C    S T E P



%% M-C   Component Inference & NMF

% UPDATE
%   C  : [F x L x K] posterior mean of components
%   W  : [F x 1 x K] array of bases
%   H  : [1 x L x K] array of contributions
%   u  : [F x L x K] prior variance of compoents

% {F x L} x [J x 1] here C = A*X/v - Ut*s/v
C = cellfun(@(A,X,v,U,S) ( reshape(A,I,J)'*X - U*S ) / v, A,X,v,U,S, 'uniformoutput', false);

% [F x L x J] cast C into array
C = cell2array(C);

% [F x L x K] posterior mean of C, note the replication into K via jk
C = u .* C(:,:,jk);

% {F x L} x [J x 1] piece of component's covariance diagonal entries
D = cellfun(@(U,Vs,v) real( diag(U*Vs) / v ), U,Vs,v, 'uniformoutput', false);

% [F x L x J] cast into array, MULTIPLY by source prior precision
D = cell2array(D) .* b;

% [F x L x K] posterior power of Components Qc = Vc_{kk,fl} + |C_{k,fl}|^2
D = u .* ( 1 - u .* D(:,:,jk) )  + C .* conj(C);

% [F x 1 x K] update bases using previous/initial H   
W = sum( bsxfun(@rdivide,D,H) ,2 ) / L;

% [1 x L x K] update activation
H = sum( bsxfun(@rdivide,D,W) ,1 ) / F;

% [F x L x K] prior variance of C
u = bsxfun(@times,W,H);
%%




%  |\  /|
%  | \/ | - X    S T E P



%% M-X   Isotropic sensor variance

% UPDATE
%   v   : {F x L} x [1 x 1] sensor noise variance, replicated on L
%   Qs  : {F x L} x [J x J] 2nd order moment of S

% {F x L} x [J x J] 2nd order moment of S
Qs = cellfun(@(Vs,S) Vs + S*S', Vs, S, 'uniformoutput', false);

% [F x 1] sensor noise variance, note that   trace(Qs*Ut) = Qs(:)'*Ut(:)
v = sum( cellfun(@(X,A,S,Qs,U)  X'*X -2*X'*reshape(A,I,J)*S + Qs(:)'*U(:), X,A,S,Qs,U), 2) / (L*I);

% {F x L} x [1 x 1] Re{} is for the linear term and deflations of tr{Qs*Ut}
v = repmat( num2cell( real(v) + 1e-7 ), 1, L );
%%




%   ____
%  |
%  |____
%  |
%  |____ - A    S T E P



%% E-A   chanel's marginal posterior

% UPDATE
%   Va  : {F x L}   x [IJ x IJ] marginal posterior covariance of vec(A)
%   A   : {F x L}   x [IJ x 1 ] marginal posterior mean of vec(A)
%   fV  : {F x L}   x [IJ x IJ] forward  covariances
%   fA  : {F x L}   x [IJ x 1 ] forward  means
%   bV  : {F x L}   x [IJ x IJ] backward covariances
%   bA  : {F x L}   x [IJ x 1 ] backward means
%   zV  : {F x L-1} x [IJ x IJ] zeta covariances (avoid subtraction in M-A)

% instantaneous statistics

% {F x L} x [IJ x IJ] instantaneous precision of A  [ used as precision ]
D = cellfun(@(Qs,v) Qs(cloneTranspose)/v .* onesKronIden, Qs, v, 'uniformoutput', false);

% {F x L} x [IJ x 1] instantaneous mean x instantaneous precision
d = cellfun(@(X,S,v) reshape( (X/v)*S', IJ,1 ), X,S,v,'uniformoutput',false);

% forward pass

% {F x L} x [IJ x IJ] initialise forward covariances
fV = [ cellfun(@(D,eV) inv(D + inv(eV)), D(:,1), eV, 'uniformoutput', false)    cell(F,L-1) ];

% {F x L} x [IJ x 1]  initialise forward means , use A(:,1) as prior means
fA =  [ cellfun(@(fV,d,eV,pA) fV * (d+eV\pA), fV(:,1), d(:,1), eV, A(:,1), 'uniformoutput', false)    cell(F,L-1) ];

for l=2:L
    % {F x L} x [IJ x IJ] forward covariances
    fV(:,l) = cellfun(@(D,fV,eV) inv(D + inv(fV+eV)),  D(:,l), fV(:,l-1), eV, 'uniformoutput', false);
    
    % {F x L} x [IJ x 1]  forward means
    fA(:,l) = cellfun(@(fV,d,fV_prev,eV,fA) fV * (d + (fV_prev+eV)\fA), fV(:,l), d(:,l), fV(:,l-1), eV, fA(:,l-1), 'uniformoutput', false);
end

% backward pass

% {F x L} x [IJ x IJ] initialise backward covariances
bV = [ cell(F,L-1)    fV(:,L) ];

% {F x L} x [IJ x 1]  initialise backward means
bA = [ cell(F,L-1)    fA(:,L) ];

% {F x L-1} x [IJ x IJ] intermediate covariances, means are equal to bA
zV = cell(F,L-1);

for l=L-1:-1:1
    % {F x L-1} x [IJ x IJ] covarinces of intermediate,l accounts for l+1
    zV(:,l) = cellfun(@(D,bV) inv(D + inv(bV)),  D(:,l+1), bV(:,l+1), 'uniformoutput' ,false);
    
    % {F x L} x [IJ x 1] backward means, they require zV not bV at l
    bA(:,l) = cellfun(@(zV,d,bV,bA) zV*(d + bV\bA),  zV(:,l), d(:,l+1), bV(:,l+1), bA(:,l+1), 'uniformoutput' ,false);
    
    % {F x L} x [IJ x IJ] backward covariances, keep zV is needed in M-A
    bV(:,l) = cellfun(@plus,zV(:,l),eV,'uniformoutput',false);
end

% marginal posterior

% {F x L} x [IJ x IJ] marginal covariances [ using the Searle Identities ]
Va = cellfun(@(fV,bV) fV*( (fV+bV)\bV ), fV, bV, 'uniformoutput' , false);

% {F x L} x [IJ x 1] marginal posterior filter means
A = cellfun(@(Va,fV,fA,bV,bA) Va * ( fV\fA + bV\bA ),  Va, fV, fA, bV, bA, 'uniformoutput', false);
%%




%  |\  /|
%  | \/ | - A    S T E P



%% M-A   process noise covariance

% UPDATE
%   eV  : {F} x [IJ x IJ] evolution covariance of vec(A)

% {F x L-1} x [IJ x IJ] invert temporarily
D = repmat(cellfun(@inv,eV,'uniformoutput',false), 1, L-1);

% {F x L-1} x [2IJ x 2IJ] joint covariance of [Afl,Afl-1]^T, D is temp
D = cellfun(@(zV,fV,D) inv( [inv(zV)+D, -D; -D, inv(fV)+D] + 1e-7 * eye(2*IJ) ), zV, fV(:,1:L-1), D, 'uniformoutput', false);

% {F x L-1} x [2IJ x 1] joint posterior means, use d as temp
d = cellfun(@(D,zV,bA,fV,fA) D * [ zV\bA ; fV\fA ],  D, zV, bA(:,2:L), fV(:,1:L-1), fA(:,1:L-1), 'uniformoutput', false);

% {F x L-1} x [2IJ x 2IJ] 2nd moment of joint filters, D as temp
D = cellfun(@(D,d) D + d*d', D, d, 'uniformoutput', false);

% {F} x [2IJ x 2IJ] sum D along the frames
D = arrayfun(@(f) sum(cat(3,D{f,:}),3), transpose(1:F), 'uniformoutput', false);

% {F} x [IJ x IJ] blksum 
eV = cellfun(@(D,Va) symtricize( D(b1,b1) - D(b2,b1) - D(b1,b2) + D(b2,b2) + Va ) / L, D, Va(:,1), 'uniformoutput', false);




fprintf('Iteration %d of %d\n',iter,maxNumIter);
end







%% compactify output (convenient dimensions)

A = permute( cell2array(A) , [1 2 4 3] );
S = cell2array(S);
W = permute( W , [1 3 2] ); 
H = permute( H , [3 2 1] );

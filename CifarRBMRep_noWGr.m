%---CifarRBM with RGB percentage and motion

%---input parameters
max_iter=1;
netName='500-500-2000';
DN.layersize   = [500,500,2000];
%----


% load('batchTrnGrCifar','batchTrnGrCifar');
% batchdata=batchTrnGrCifar;

load('batchTrnGrCifarEx','batchTrnGrCifarEx');
batchdata=batchTrnGrCifarEx;

% load('batchcolorTr','batchcolorTr');
% colorVect=batchcolorTr;

% load('batchMoTr','batchMoTr');
% motionVect=batchMoTr;


% load('batchTrnLblCrCifar','batchTrnLblCrCifar');
% load('batchTrnLblFnCifar','batchTrnLblFnCifar');
%
% load('cifarGrTst','cifarGrTst');
% load('coarseLblCifarTst','coarseLblCifarTst');
% load('fineLblCifarTst','fineLblCifarTst');

% Cl=100;


% netName='500-500-2000';
% DN.layersize   = [500,500,2000];
% a=.2;
% b=.1;
% c=.01;
h=.03;
% g=.04;
% d=.05;
% e=.07;
% f=.02;
DN.nlayers     = length(DN.layersize);
DN.maxepochs   = 100;
DN.batchsize   = 100;
epsilonw_GPU   = gpuArray(h);
epsilonvb_GPU  = gpuArray(h);
epsilonhb_GPU  = gpuArray(h);
% D=.00002;
% A=.001;
% B=.0001;
% C=.00001;
E=.00005;

weightcost_GPU = gpuArray(E);

max_iter=1;
for iter=1:max_iter
    iter
    DN.err = zeros(DN.maxepochs, DN.nlayers, 'single');
    
    for layer = 1:DN.nlayers
        
        fprintf(1,'Training layer %d...\n', layer);
        if layer == 1
            data_GPU = gpuArray(single(batchdata));
        else
            data_GPU  = batchposhidprobs;
        end
        
        % initialize weights and biases
        numhid  = DN.layersize(layer);
        [numcases numdims numbatches] = size(data_GPU);
        numcases_GPU     = gpuArray(numcases);
        vishid_GPU       = gpuArray(0.1*randn(numdims, numhid, 'single'));
        hidbiases_GPU    = gpuArray(zeros(1,numhid, 'single'));
        visbiases_GPU    = gpuArray(zeros(1,numdims, 'single'));
        vishidinc_GPU    = gpuArray(zeros(numdims, numhid, 'single'));
        hidbiasinc_GPU   = gpuArray(zeros(1,numhid, 'single'));
        visbiasinc_GPU   = gpuArray(zeros(1,numdims, 'single'));
        batchposhidprobs = gpuArray(zeros(DN.batchsize, numhid, numbatches, 'single'));
        
        for epoch = 1:DN.maxepochs
            
            errsum = 0;
            for mb = 1:numbatches
                data_mb = data_GPU(:, :, mb);
                % learn an RBM with 1-step contrastive divergence
                rbm_GPU;
                errsum = errsum + err;
                if epoch == DN.maxepochs
                    batchposhidprobs(:, :, mb) = poshidprobs_GPU;
                end
            end
            DN.err(epoch, layer) = errsum;
            
        end
        DN.L{layer}.hidbiases  = gather(hidbiases_GPU);
        DN.L{layer}.vishid     = gather(vishid_GPU);
        DN.L{layer}.visbiases  = gather(visbiases_GPU);
        
    end
    
    CifarRBM{iter}=DN;

end
saveName=sprintf('CifarNWRBM_DN_hB_L3Gr%s_maxEp%d',netName,epoch);

save(saveName,'CifarRBM');

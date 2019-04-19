mot=0; %motion information

%---input parameters
max_iter=1;
netName='600-1000-2000';
DN.layersize   = [600,1000,2000];
%----


h=.03;

DN.nlayers     = length(DN.layersize);
DN.maxepochs   = 200;
DN.batchsize   = 10;
epsilonw_GPU   = gpuArray(h);
epsilonvb_GPU  = gpuArray(h);
epsilonhb_GPU  = gpuArray(h);

E=.00005;

weightcost_GPU = gpuArray(E);

load('batchTrnGrCalt','batchTrnGrCalt');
batchdata=batchTrnGrCalt;

% load('batchMoTrAnC','batchMoTrAnC');

for iter=1:max_iter
    iter
    DN.err = zeros(DN.maxepochs, DN.nlayers, 'single');
    for layer = 1:DN.nlayers
        fprintf(1,'Training layer %d...\n', layer);
        if layer == 1
            data_GPU = gpuArray(single(batchdata));
        elseif layer == 2
            data_GPU  = batchposhidprobs;
        elseif layer == 3
            data_GPU  = batchposhidprobs;
            if mot == 1
                data_GPU = horzcat(data_GPU,batchMoTrArt);
            end
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
    CaltRBM{iter}=DN;
end
if mot==1
  saveName=sprintf('CaltNWGrRBM_DN_hBAnPl%s_maxEp%d_maxIt%d',netName,epoch,max_iter);
else  
saveName=sprintf('CaltNWGrRBM_DN_hBAnPl%s_maxEp%d_maxIt%d',netName,epoch,max_iter);
end
save(saveName,'CifarRBM');


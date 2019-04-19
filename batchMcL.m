

load('stim','nS');
batchsize=89;
totnum = size(nS, 1);
randomorder = randperm(totnum);
numbatches   = round(totnum/batchsize);

numdims      = size(nS, 2);
batchdata    = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, numbatches);
for b=1:numbatches
    batchdata(:,:,b)  = nS(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;

save('McVisBatch','batchdata','numbatches');
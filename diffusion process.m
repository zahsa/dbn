x=0:0.1:10;
y=gaussmf(x,[2 5]);
plot(x,y)
xlabel('gaussmf, P=[2 5]')




mu = [0 0]; 
SIGMA = [.9 .4; .4 .3]; 
X = mvnrnd(mu,SIGMA,10); 
X2=[-.1:.01:.1;-1:.1:1]';
p = mvnpdf(X2,mu,SIGMA); 

surf(X2,p)
mesh(X2,p)
plot(X2(:,2),p)

[Xx,Yy] = meshgrid(-3:.125:3);
Z = peaks(Xx,Yy);
mesh(Xx,Yy,Z);
surf(Xx,Yy)

c=cov(X);
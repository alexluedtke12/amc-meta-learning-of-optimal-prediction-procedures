
library(flam)
library(MASS)
library(matlib)
library(parallel)

# https://r.789695.n4.nabble.com/Suppressing-output-e-g-from-cat-td859876.html
quiet <- function(x) { 
  sink(tempfile()) 
  on.exit(sink()) 
  invisible(force(x)) 
} 

# flam in which the sum M of the increments is known in advance,
# and the regularization parameter is chosen so that the empirical
# sum of the increments is (approximately) equal to M
flam_knownM_est = function(w,y,w_tilde,M=10,...){
	require(flam)
	alpha = 1
	n.lambda = 100
	flam_out = flam(w,y,alpha.seq=alpha,n.lambda=n.lambda)
	flam_out$theta.hat.list[[1]]

	var_norm_hats = sapply(1:n.lambda,function(lambda.ind){
		tmp = flam_out$theta.hat.list[[lambda.ind]]
		for(i in 1:ncol(w)){
			tmp[,i] = tmp[order(tmp[,i]),i]
		}
		sum(abs(tail(tmp,n=-1)-head(tmp,n=-1)))
	})

	ind = which.min(abs(var_norm_hats-M))
	predict(flam_out,w_tilde,lambda=flam_out$all.lambda[ind],alpha=alpha)
}


standardize_ws = function(w,w_tilde){
	n = nrow(w)
	n_tilde = nrow(w_tilde)

	mu_w = colMeans(w)
	sigma_w = sqrt(colMeans(w^2)-mu_w^2)

	w = (w - matrix(mu_w,nrow=n,ncol=length(mu_w),byrow=TRUE))/matrix(sigma_w,nrow=n,ncol=length(mu_w),byrow=TRUE)
	w_tilde = (w_tilde - matrix(mu_w,nrow=n_tilde,ncol=length(mu_w),byrow=TRUE))/matrix(sigma_w,nrow=n_tilde,ncol=length(mu_w),byrow=TRUE)

	return(list(w=w,w_tilde=w_tilde))
}

# evaluate and save the (sum of the component) variation norms in the simulation in Petersen et al.
var_norms = sapply(1:4,function(scenario){
  dat = quiet(flam::sim.data(1e5,scenario,zerof=0))
  
  for(i in 1:4){
    dat$theta[,i] = dat$theta[order(dat$x[,i]),i]
  }
  
  diffs = abs(tail(dat$theta,n=-1)-head(dat$theta,n=-1))
  colSums(diffs)
})

gen_data_flam = function(n,scenario,p=10,n_tilde=1000,M=10,s=4,...){
  # training and testing inds
  train_inds = 1:n
  test_inds = (n+1):(n+n_tilde)
  
  if(p<4){
  	stop("gen_data_flam requires p>=4.")
  }

  # simulate data
  dat = quiet(flam::sim.data(n+n_tilde,scenario,zerof=p-4))
  # save predictors
  w_tilde = dat$x[test_inds,]
  w = dat$x[train_inds,]

  if(s<4){
  	keep_inds = sort(sample(1:4,s))
  	remove_inds = setdiff(1:4,keep_inds)
  	# only keep s (random) signal components
  	dat$theta[,remove_inds] = 0
  } else {
  	keep_inds = 1:4
  }

  # rescale theta so that the sum of the component variation norms is M
  dat$theta = dat$theta * M/sum(var_norms[keep_inds,scenario])
  
  # outcome and regression
  y = rowSums(dat$theta[train_inds,]) + rnorm(n)
  regfun_tilde = rowSums(dat$theta[test_inds,])
  
  return(c(standardize_ws(w,w_tilde),list(y=y,regfun_tilde = regfun_tilde,theta=dat$theta)))
}

# Evaluates risk of an estimator, and saves the datasets into wd (if wd is not NA)
eval_risk = function(predictor,n,s,gen_data,wd=NA,predictor_name=NA,num_mc=1000,mc.cores=1){
	require(parallel)
	if(!dir.exists(wd)){ dir.create(wd) }
	losses = unlist(mclapply(0:(num_mc-1),function(i){
		dat = gen_data(n,s=s)
		if(!is.na(wd)){
			w_file = file.path(wd,paste0("w_n",n,"_s",s,"_mcrep",i,".csv"))
			w_tilde_file = file.path(wd,paste0("w_tilde_n",n,"_s",s,"_mcrep",i,".csv"))
			y_file = file.path(wd,paste0("y_n",n,"_s",s,"_mcrep",i,".csv"))
			regfun_file = file.path(wd,paste0("regfun_n",n,"_s",s,"_mcrep",i,".csv"))
			write.csv(dat$w,file=w_file,row.names=FALSE)
			write.csv(dat$w_tilde,file=w_tilde_file,row.names=FALSE)
			write.csv(cbind(dat$y),file=y_file,row.names=FALSE)
			write.csv(cbind(dat$regfun),file=regfun_file,row.names=FALSE)
		}
		return((predictor(dat$w,dat$y,dat$w_tilde) - dat$regfun)^2)
			},mc.cores=mc.cores))
	if(!is.na(wd) & !is.na(predictor_name)){
		write.csv(losses,file=file.path(wd,paste0("losses_",predictor_name,"_n",n,"_s",s,".csv")),row.names=FALSE)
	}
	return(list(mse=mean(losses),se=sd(losses)/sqrt(num_mc)))
}


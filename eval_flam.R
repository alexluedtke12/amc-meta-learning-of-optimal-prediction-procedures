source("funs.R")

set.seed(1)

# Number of training data sets
num_mc = 2000
# Sample sizes
n_vals = c(100,500)
# Sparsity levels
s_vals = c(1,5)
# Comparator
predictor = flam_knownM_est
predictor_name = "flam"
# Main working directory
main_wd = "./"
# number of cores to use when parallelizing
num.cores=1

# Evaluate risk in Petersen scenarios for flam


if(!dir.exists(file.path(main_wd,"sim_data"))){ dir.create(file.path(main_wd,"sim_data")) }

mat = NULL
for(s in s_vals){
	for(scenario in 1:4){
		for(n in n_vals){
			print(paste0("(s,scenario,n) = (",s,",",scenario,",",n,")"))
			curr_gen_data = function(n,p=10,n_tilde=1000,M=10,s=s,...){
				gen_data_flam(n=n,scenario=scenario,p=p,n_tilde=n_tilde,M=M,s=s,...)
			}
			risks = eval_risk(predictor=predictor,n=n,s=s,gen_data=curr_gen_data,num_mc=num_mc,wd=file.path(main_wd,paste0("sim_data/flam_",scenario)),predictor_name=predictor_name,mc.cores=num.cores)
			mat = rbind(mat,data.frame(scenario=scenario,est=predictor_name,n=n,s=s,mse=risks$mse,se=risks$se))
		}
	}
	write.csv(mat,file=file.path(main_wd,"sim_data/",paste0(predictor_name,"_performance.csv")),row.names=FALSE)
}

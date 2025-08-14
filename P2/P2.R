#Step 1: Load packages
pkgs <- c("tidyverse","mclust","caret","pROC")
install.packages(setdiff(pkgs, rownames(installed.packages())), dep = TRUE)
lapply(pkgs, library, character.only = TRUE)

#Step 2: initialize helpers
scalar_or_NA <- function(x) if (length(x)) x[1] else NA
model_label  <- function(obj) dplyr::coalesce(scalar_or_NA(obj$modelName),
                                              scalar_or_NA(attr(obj$EDDA,"modelName")))
metric_pack  <- function(tp,fp,fn,tn){
  prec <- ifelse(tp+fp==0, NA, tp/(tp+fp))
  rec  <- ifelse(tp+fn==0, NA, tp/(tp+fn))
  F1   <- ifelse(is.na(prec)|is.na(rec)|prec+rec==0, NA,
                 2*prec*rec/(prec+rec))
  spec <- ifelse(tn+fp==0, NA, tn/(tn+fp))
  c(F1=F1, Sens=rec, Spec=spec)
}

#Step 3: load data
data_path <- "C:/Users/saras/OneDrive/Desktop/Dissertation/Dataset/anonymisedData/"

studentVle  <- read_csv(file.path(data_path,"studentVle.csv"))
studentInfo <- read_csv(file.path(data_path,"studentInfo.csv"))
assess      <- read_csv(file.path(data_path,"assessments.csv"))
studAssess  <- read_csv(file.path(data_path,"studentAssessment.csv"))
vle         <- read_csv(file.path(data_path,"vle.csv"))

#Step 4: learners & slice boundaries 
bbb_students <- studentInfo %>%
  filter(code_module=="BBB", code_presentation=="2013J") %>%
  select(id_student)

bbb_assess <- assess %>%
  filter(code_module=="BBB", code_presentation=="2013J", !is.na(date)) %>%
  arrange(date)

cutoffs <- sort(unique(c(0, bbb_assess$date,
                         max(studentVle$date, na.rm = TRUE))))

#Step 5: rare-feature pruning 
raw_types <- studentVle %>%
  semi_join(bbb_students, by = "id_student") %>%
  inner_join(vle %>% select(id_site, activity_type), by = "id_site") %>%
  group_by(activity_type) %>%
  summarise(prop = n_distinct(id_student)/n_distinct(bbb_students$id_student),
            .groups="drop")

rare_types <- raw_types %>% filter(prop < 0.01) %>% pull(activity_type)

#Step 6: behavioural features 
bbb_vle <- studentVle %>%
  semi_join(bbb_students, by="id_student") %>%
  inner_join(vle %>% select(id_site, activity_type), by="id_site") %>%
  filter(!activity_type %in% rare_types) %>%               # ← prune rare
  mutate(interval = findInterval(date, cutoffs, rightmost.closed = TRUE))

feat <- bbb_vle %>%
  group_by(id_student,interval,activity_type) %>%
  summarise(clicks   = log1p(sum(sum_click)),              # ← log1p
            sessions = log1p(n()),
            .groups  = "drop") %>%
  pivot_wider(names_from  = activity_type,
              values_from = c(clicks,sessions),
              names_glue  = "{.value}_{activity_type}",
              values_fill = 0)

## add previous-slice features
prev <- feat %>% mutate(interval = interval + 1) %>%
  rename_with(~ paste0(.x,"_prev"), -c(id_student,interval))

feat <- feat %>% left_join(prev, by = c("id_student","interval")) %>%
  mutate(across(where(is.numeric), ~ replace_na(.x, 0)))

#Step 7: target label 
flags <- studAssess %>%
  semi_join(bbb_students, by="id_student") %>%
  inner_join(bbb_assess %>% select(id_assessment,date), by="id_assessment") %>%
  mutate(interval = findInterval(date, cutoffs, rightmost.closed = TRUE)) %>%
  group_by(id_student,interval) %>% summarise(submitted=1,.groups="drop")

data_all <- feat %>% left_join(flags, by = c("id_student","interval")) %>%
  mutate(submitted = replace_na(submitted,0),
         label     = factor(ifelse(submitted==1,0,1), levels=c(0,1)))

#Step 8: 10× evaluation per slice 
set.seed(42)
results <- vector("list", 6)           # one slot per time slice (t = 1‥6)

for (t in 1:6) {
  cat("Slice", t, "… ")
  
  slice <- data_all %>% filter(interval == t)
  if (nrow(slice) < 10 || length(unique(slice$label)) < 2) {
    cat("skipped\n"); next
  }
  
  X <- slice %>% select(-id_student, -interval, -submitted, -label) %>% as.data.frame()
  y <- slice$label
  
  run_out <- matrix(NA, nrow = 10, ncol = 5,
                    dimnames = list(NULL, c("F1","Sens","Spec","AUC","Model")))
  
  for (run in 1:10) {
    
    ## 60/40 stratified split 
    idx  <- createDataPartition(y, p = 0.60, list = FALSE)
    X_tr <- X[idx , , drop = FALSE];  y_tr <- y[idx ]
    X_te <- X[-idx, , drop = FALSE];  y_te <- y[-idx]
    if (length(unique(y_te)) < 2) next                 # need both classes in TEST
    
    ## bootstrapped up-sampling of minority 
    ups  <- upSample(X_tr, y_tr, yname = "label")
    X_tr <- ups %>% select(-label);  y_tr <- ups$label
    
    ## drop zero-var cols  
    keep <- apply(X_tr, 2, var) > 0
    X_tr <- X_tr[ , keep, drop = FALSE]
    X_te <- X_te[ , keep, drop = FALSE]
    
    scaler <- preProcess(X_tr, method = c("center","scale"))
    X_tr   <- predict(scaler, X_tr)
    X_te   <- predict(scaler, X_te)
    
    ## fit EDDA
    gmm <- tryCatch(
      MclustDA(X_tr, class = y_tr, modelNames = c("EEE","EEV")),
      error = function(e) NULL
    )
    if (is.null(gmm)) next                              # singular fit
    
    ## optimise prob-threshold on ORIGINAL
    post_orig <- predict(gmm, X[idx, , drop = FALSE])$z[ , "1"]
    if (all(post_orig == 0) || all(is.na(post_orig))) next   # degenerate fit
    y_orig <- y[idx]
    
    thr_seq  <- seq(0.05, 0.95, by = 0.01)
    train_F1 <- sapply(thr_seq, function(th) {
      y_hat <- factor(ifelse(post_orig >= th, 1, 0), levels = c(0,1))
      tbl   <- table(y_orig, y_hat)
      metric_pack(tbl["1","1"], tbl["0","1"], tbl["1","0"], tbl["0","0"])["F1"]
    })
    if (all(is.na(train_F1))) next                       # nothing usable
    opt_thr <- thr_seq[ which.max(train_F1) ]
    
    ## evaluate on TEST with chosen threshold 
    post_te <- predict(gmm, X_te)$z[ , "1"]
    ok_row  <- !is.na(post_te)                         
    if (sum(ok_row) < 2) next                        
    post_te <- post_te[ok_row];  y_cmp <- y_te[ok_row]
    
    y_hat <- factor(ifelse(post_te >= opt_thr, 1, 0), levels = c(0,1))
    tbl   <- table(y_cmp, y_hat)
    
    mp <- metric_pack(tbl["1","1"], tbl["0","1"], tbl["1","0"], tbl["0","0"])
    auc_val <- tryCatch(
      auc(roc(as.numeric(y_cmp), post_te)),
      error = function(e) NA_real_
    )
    
    run_out[run, ] <- c(mp, auc_val, model_label(gmm))
  }  
  
  ## aggregate slice results 
  valid <- !is.na(run_out[ , "F1"])
  if (!any(valid)) { cat("no valid run\n"); next }
  
  res   <- run_out[valid, , drop = FALSE]
  major <- res[,"Model"] |> na.omit() |> table() |> sort(decreasing = TRUE) |> names() |> scalar_or_NA()
  
  results[[t]] <- data.frame(
    Slice   = t,
    Model   = major,
    F1_M    = mean(res[ , "F1"]),   F1_SD   = sd(res[ , "F1"]),
    Sen_M   = mean(res[ , "Sens"]), Sen_SD  = sd(res[ , "Sens"]),
    Spec_M  = mean(res[ , "Spec"]), Spec_SD = sd(res[ , "Spec"]),
    AUC_M   = mean(res[ , "AUC"]),  AUC_SD  = sd(res[ , "AUC"]),
    row.names = NULL
  )
  cat("done\n")
} 

results_df <- bind_rows(results)
cat("\nFinal reproduction of Table 2:\n")
print(results_df, digits = 3)

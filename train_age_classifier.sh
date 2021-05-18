python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 4000 "1/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 4000 "2/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 4000 "3/"

python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 1000 "1/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 1000 "2/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 1000 "3/"
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 2000 "1/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 2000 "2/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 2000 "3/"
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 3000 "1/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 3000 "2/"
rm -r /cluster/home/reisedom/data_german/cached_age
python ./train_classifier_age.py "config/config_reisedom_german_age_max_samples.json" 3000 "3/"
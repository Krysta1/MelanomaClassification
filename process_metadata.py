import pandas as pd
import numpy as np


path = "/home/xinsheng/skinImage/data/jpeg-melanoma-256/"


def main(df, save_path, meta):

    if meta:
        df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].map({'lateral torso': 'torso', 'posterior torso': "torso",'anterior torso': "torso", "torso": "torso", "lower extremity": "lower extremity", "upper extremity": "upper extremity", "head/neck": "head/neck", "palms/soles": "palms/soles", "oral/genital": "oral/genital"})
        # df['anatom_site_general_challenge'].fillna('torso', inplace=True)
    # generate one hot code for 'anatom_site_general_challenge'
    dummies = pd.get_dummies(df['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')
    df = pd.concat([df, dummies], axis=1)
    
    # map string values to integer
    try:
        df['diagnosis'] = df['diagnosis'].map({'unknown': 2, 'seborrheic keratosis': 2, 
                                                            'lentigo NOS': 2, 'lichenoid keratosis': 2, 'solar lentigo': 2, 
                                                            'atypical melanocytic proliferation': 2, 'cafe-au-lait macule': 2, 
                                                            'nevus': 1, 'melanoma': 0
                                                            })
    except:
        pass
    df['sex'] = df['sex'].map({"male": 1, "female": 0})
    df['sex'] = df['sex'].fillna(-1)
    # count numbers of images for each patient
    df['n_images'] = df.patient_id.map(df.groupby(['patient_id']).image_name.count())
    df['n_images'] = df['n_images'].fillna(df['n_images'].mean())
    print(f"after fillna the Nan nums is {df['n_images'].isna().sum()}")
    
    # normolize age by divide the biggest age in the dataset
    print(f"before fillna the Nan nums if {df['age_approx'].isna().sum()}")
    average = df['age_approx'].mean()
    print(f"average age is {average}")
    df['age_approx'].fillna(average, inplace=True)
    print(f"after fillna the Nan nums is {df['age_approx'].isna().sum()}")
    df['age_approx'] /= 90.0

    # save to local. 
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    train_df = pd.read_csv("/home/xinsheng/skinImage/melanoma-external-malignant-256/train_concat.csv")
    test_df = pd.read_csv(path + "test.csv")
    main(train_df, "./data/train-extra-jpeg-256.csv", True)
    # main(test_df, "./data/test-jpeg-256.csv")

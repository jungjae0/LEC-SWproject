import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def main():
    data_dir = r"D:\개인\2024SW융합프로젝트\99_data\pudina"



    df = pd.DataFrame()
    for root, dirs, files in os.walk(data_dir):
        if 'Multiple' not in root and 'Sunlight' not in root:
            count = len(files)
            label = root.split('\\')[-2]
            if count != 0:
                each = pd.DataFrame(files, columns=['filename'])
                each['filepath'] = root + '\\' + each['filename']
                each['label'] = label

                df = pd.concat([df, each], ignore_index=True)

    value_counts = df['label'].value_counts()

    value_counts.plot(kind='bar', color='blue')

    plt.title('Value Counts of Labels')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


    df['label'] = df['label'].map({'Dried': 2, 'Spoiled':1, 'Fresh': 0})


    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_data, val_data = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    train_data['split'] = 'train'
    test_data['split'] = 'test'
    val_data['split'] = 'val'

    df = pd.concat([val_data, test_data, train_data])

    # train_data.to_csv('../train_dataset.csv', index=False)
    # test_data.to_csv('../test_dataset.csv', index=False)
    # val_data.to_csv('../val_dataset.csv', index=False)

    counts = df.groupby(['split', 'label']).size().reset_index()
    counts = counts.pivot(index='label', columns='split', values=0)
    ax =  counts.plot(kind="bar", stacked=True, figsize=(15, 10))

    for c in ax.containers:
        labels = [int(x.get_height()) for x in c]
        ax.bar_label(c, labels=labels, label_type='center')

    label_mapping = {0: 'Fresh', 1: 'Spoiled', 2: 'Dried'}
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([label_mapping[i] for i in ax.get_xticks()])

    plt.xticks(rotation=45)
    plt.savefig('dataset_split.png')






if __name__ == '__main__':
    main()
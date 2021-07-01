import librosa,glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle
from keras import optimizers,losses,activations,models,Sequential,regularizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense,Input,BatchNormalization,Dropout,Convolution2D,MaxPooling2D,GlobalMaxPooling2D,Conv2D
from keras.optimizers import adam
from keras.layers import Flatten,ReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from librosa.effects import time_stretch,pitch_shift
import soundfile as sf
from NN.Tomofun_狗音辨識.audio_classification.resnet18 import ResNet


#https://github.com/CVxTz/audio_classification/blob/master/code/keras_cnn_mel.py
def emphasized_filter(x):
    ##pre-emphasis y(t)=x(t)-ax(t-1)
    emphasis_coff=0.98
    emphasized_x = np.append(x[0],x[1:]-emphasis_coff*x[:-1])
    return emphasized_x
def audio_preprocessing_mel_T(audio,sample_rate=8000):
    stft = np.abs(librosa.core.stft(audio))
    mel_sec = librosa.feature.melspectrogram(S=stft**2,sr=sample_rate,n_mels=320)
    mel_db = (librosa.power_to_db(mel_sec,ref=np.max)+40)/40 ##讓特徵圖更明顯
    ##視覺化
    # librosa.display.specshow(mel_db, y_axis='mel', x_axis='time')
    # plt.show()
    return mel_db.T
def data_time_stretch(path,count,data,sr=8000):
    ts = time_stretch(data,rate=1.1)
    sf.write(f'{path}/train_0{count}.wav',ts,sr)
def data_white_noise(path,count,data,sr=8000):
    wn = np.random.randn(len(data))
    data_wn = data+0.005*wn
    sf.write(f'{path}/train_0{count}.wav',data_wn,sr)
def data_pitch_change(path,count,data,sr=8000):
    pc = pitch_shift(data,sr,n_steps=1.1,bins_per_octave=2)
    sf.write(f'{path}/train_0{count}.wav', pc, sr)
def data_argumentation():
    count = 1201
    path = '../Tomofun_datasets/train/'
    t = sorted(train_files)
    tmp1=pd.read_csv('meta_train.csv')
    for index_orifile in range(1200):
        data = librosa.load(t[index_orifile],sr=8000,mono=True)[0].astype(np.float32)
        data_pitch_change(path,count,data)
        r = f'train_0{count}'
        tmp1.at[index_orifile,'Filename'] = r
        print(count)
        count += 1
    tmp2=pd.read_csv('meta_train.csv')
    for index_orifile in range(1200):
        data = librosa.load(t[index_orifile], sr=8000, mono=True)[0].astype(np.float32)
        data_white_noise(path,count,data)
        r = f'train_0{count}'
        tmp2.at[index_orifile,'Filename'] = r
        print(count)
        count += 1
    tmp0=pd.read_csv('meta_train.csv')
    final = pd.concat([tmp0,tmp1,tmp2], axis=0)
    print(final)
    final.to_csv('train_afterDA.csv',index=False)
    return final
def load_audio_file(path,input_length=40000,istest=True):
    #輸出是一個tumple，只取data部分
    data = librosa.load(path,sr=8000,mono=True)[0].astype(np.float32)
    if len(data) < input_length:
        ##padding with zeros
        s = np.zeros((input_length,), dtype=np.int16)
        s[:len(data)] = data
        data = s.astype(np.float32)
    elif len(data) > input_length:
        s = data[:input_length]
        data = s.astype(np.float32)
    if istest:
        data = emphasized_filter(data)
    data = audio_preprocessing_mel_T(data)
    return data
def file_to_label():
    file_path = '../Tomofun_datasets/train/'
    labeled_files = {}
    for filename,label in zip(train_labels.Filename.values,train_labels.Label.values):
        labeled_files[file_path+filename+'.wav'] = label
    return labeled_files
def get_model_mel():
    nclass = len(list_labels)
    input = Input(shape=(79,320,1))
    norm_input = BatchNormalization(momentum=0.8)(input)
    img = Convolution2D(16,kernel_size=(3,7),activation=activations.relu)(norm_input)
    img = Convolution2D(16,kernel_size=(3,7),activation=activations.relu)(img)
    img = MaxPooling2D(pool_size=(3,7))(img)
    img = Dropout(rate=0.1)(img)

    img = Convolution2D(32, kernel_size=3, activation=activations.relu)(img)
    img = Convolution2D(32, kernel_size=3, activation=activations.relu)(img)
    img = MaxPooling2D(pool_size=(3, 3))(img)
    img = Dropout(rate=0.1)(img)

    img = Convolution2D(128,kernel_size=3,activation=activations.relu)(img)
    img = GlobalMaxPooling2D()(img)
    img = Dropout(rate=0.1)(img)

    dense = BatchNormalization(momentum=0.8)(Dense(128,activation=activations.relu)(img))
    dense = BatchNormalization(momentum=0.8)(Dense(128,activation=activations.relu)(dense))
    dense = Dense(nclass,activation=activations.softmax)(dense)

    model = models.Model(input,dense)
    opt = optimizers.Adam()
    model.compile(optimizer=opt,loss=losses.sparse_categorical_crossentropy,metrics=['acc'])
    model.summary()
    return model
def ResBet18():
    n_class = len(list_labels)
    model=ResNet(layer_dim=[2,2,2,2],n_class=n_class)
    model.build(input_shape=(79,320,1))
    model.summary()
    opt = optimizers.Adam()
    model.compile(optimizer=opt,loss=losses.sparse_categorical_crossentropy,metrics=['acc'])
    return model
def get_model_():
    nclass = len(list_labels)
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(1, 3), padding='same', input_shape=(79,320,1)))
    model.add(ReLU())
    model.add(Conv2D(16, kernel_size=(1, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(16, kernel_size=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(16, kernel_size=(3, 1), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(16, kernel_size=(3, 1), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(16, kernel_size=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(128, kernel_size=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(128, kernel_size=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Dense(nclass, kernel_regularizer=regularizers.l2(0.01), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4, decay=1e-6), metrics=['acc'])
    model.summary()
    return model
def get_truth():
    temp = train_labels.set_index('Filename')
    valfile_sorted = sorted(val_files)
    ground_truth=[]
    for valfile in valfile_sorted:
        t = valfile[-15:-4]
        ground_truth.append(temp.loc[t, 'Label'])
    return ground_truth
def confusmatrix(predicted):
    ground_truth=get_truth()
    predicted_set = predicted.set_index('Filename')
    valfile_sorted = sorted(val_files)
    predict=[]
    for valfile in valfile_sorted:
        t = valfile[-15:-4]
        predict.append(predicted_set.loc[t,'Label'])
    matrix = confusion_matrix(ground_truth,predict)
    fig, ax=plot_confusion_matrix(conf_mat=matrix,colorbar=True,class_names=class_name,show_normed=True,show_absolute=False,figsize=(12,12))
    plt.savefig('confusion_matrix.png')
    report = classification_report(ground_truth,predict)
    print(report)
def AUC_ROC_curve(truth,pred):
    ###Disply AUC-ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_name)):
        fpr[i], tpr[i], _ = roc_curve(truth, pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(len(class_name)):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('AUREOC.png')
def plot_loss_acc(history):
    f = open('loss_acc.txt', 'w')
    # Displaying loss values
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoches = range(1, len(train_loss) + 1)
    plt.plot(epoches, train_loss)
    plt.plot(epoches, val_loss)
    plt.xlabel('Epoches')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'val_loss'])
    plt.savefig('epoche_loss.png')
    plt.close()
    f.write(f'train_loss:{train_loss[-1]}\n')
    print('loss:', history.history['loss'][-1])
    f.write(f'val_loss:{val_loss[-1]}\n')
    print('val_loss:', history.history['val_loss'][-1])
    # Displaying accuracy scores
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epoches, train_acc)
    plt.plot(epoches, val_acc)
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(['train_acc', 'val_acc'])
    plt.savefig('epoche_accuracy.png')
    plt.close()
    f.write(f'train_acc:{train_acc[-1]}\n')
    print('acc:', history.history['acc'][-1])
    f.write(f'val_acc:{val_acc[-1]}\n')
    print('val_acc:', history.history['val_acc'][-1])
def chunker(seq,size):
    return (seq[pos:pos+size] for pos in range(0,len(seq),size))
def train_generator(list_files,batch_size = 64):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files,size=batch_size):
            batch_label = [file_to_int[fpath] for fpath in batch_files]
            batch_label = np.array(batch_label)
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:,:,:,np.newaxis]
            yield batch_data,batch_label
def training():
    history = model.fit_generator(train_generator(tr_files, batch_size=batch_size), steps_per_epoch=len(tr_files) // batch_size,
                                  epochs=50,validation_data=train_generator(val_files), validation_steps=len(val_files) // batch_size,
                                max_queue_size=60,callbacks=[ModelCheckpoint('../baseline_cnn_mel.h5', monitor='val_acc', save_best_only=True),
                                    EarlyStopping(patience=10, monitor='val_acc',mode='max')])
    model.save_weights("cnn_model.h5")
    # model.save_weights("cnn_model_2.h5")
    plot_loss_acc(history)

def predicting(predict_files,istest=True):
    bag=3
    array_preds=0
    for i in tqdm(range(bag)):
        list_predict = []
        for batchfiles in tqdm(chunker(predict_files,size=batch_size),total=len(predict_files)//batch_size):
            batch_data = [load_audio_file(fpath,istest=istest) for fpath in batchfiles]
            batch_data = np.array(batch_data)[:,:,:,np.newaxis]
            preds = model.predict(batch_data).tolist()
            list_predict+=preds
        array_preds+=np.array(list_predict)/bag
    array_preds = np.array(array_preds)
    csv = pd.DataFrame(array_preds,columns=class_name)
    df = pd.DataFrame(predict_files, columns=["Filename"])
    df=df.Filename.apply(lambda x: x.split("/")[-1][:-4])
    if not istest:
        top = [array_preds[i].argmax() for i in range(len(array_preds))]
        c=pd.DataFrame(np.array(top),columns=['Label'])
        pre = pd.concat([df,c],axis=1)
        confusmatrix(pre)
    concat = pd.concat([df,csv],axis=1)
    concat.to_csv('submission.csv',index=False)
if __name__ == '__main__':
    # train_label = pd.read_csv('./meta_train.csv')
    train_labels = pd.read_csv('train_afterDA.csv')
    train_files = glob.glob('../Tomofun_狗音辨識/Tomofun_datasets/train/*.wav')
    test_files = sorted(glob.glob('../Tomofun_狗音辨識/Tomofun_datasets/test/*.wav'))
    class_name = ['Barking','Howling','Crying','COSmoke','GlassBreaking','Other']
    list_labels = sorted(list(set(train_labels.Label.values)))
    label_to_int = {k:v for v,k in enumerate(list_labels)}
    file_to_lable = file_to_label()
    file_to_int = {k:label_to_int[v] for k,v in file_to_lable.items()}
    tr_files,val_files = train_test_split(sorted(train_files),test_size=0.1,random_state=42)
    batch_size = 64
    model = get_model_mel()
    #model = ResBet18()
    # data_argumentation()
    training()
    model.load_weights("baseline_cnn_mel.h5")
    predicting(test_files,istest=True)
    df=pd.read_csv('../submission.csv')
    add=pd.read_csv('../sample_submission.csv', )
    add=add[10000:][:]
    final = pd.concat([df,add],axis=0)
    final.to_csv('report.csv',index=False)

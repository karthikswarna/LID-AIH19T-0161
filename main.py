import os
import datetime
#from flask_socketio import SocketIO, send
import tkinter as tk
import time
from app import app
from flask import flash, request, redirect, render_template, Flask
from werkzeug.utils import secure_filename
from data_upload import dataprepocess, datapreprocess_off
#import CRNN
import onnx
import onnxruntime as ort
import numpy as np
#import gui
ALLOWED_EXTENSIONS = set(['wav'])
#model = CRNN.CRNN()
#model.load_state_dict(torch.load("weights/TrialRunWeights.pth", map_location='cpu'))
#model.train(False)
sess = ort.InferenceSession("weights/pT12.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

sessOn = ort.InferenceSession("weights/online.onnx")
inName = sessOn.get_inputs()[0].name
labName = sessOn.get_outputs()[0].name

lang_list = ["TAMIL", "GUJARATI", "MARATHI", "HINDI", "TELUGU"]

#from OpenSSL import SSL
#context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
#context = SSL.Context(OP_NO_SSLv3)
#context.use_privatekey_file('server.key')
#context.use_certificate_file('server.crt')


def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];

        list1.remove(max1);
        final_list.append(j)
    return final_list

def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
        return render_template('upload.html', loaded = 0)

#@app.route('/server', methods=['POST'])
#def open_gui():
#       flash("Live recorder started")
#       # main = tk.Tk()
#       # app = gui.App(main)
#       gui.main()
#       flash("Live recorder closed")
#       return redirect('/')

j = 0
@app.route('/messages_2', methods = ['POST'])
def api_message_2():
        global j
        print("trial 1")
        x = datetime.datetime.now()
        
        f = open('./recordings_marathi/'+str(x)+ '_file_new.wav', 'wb')
        print("trial 2")
        path = './recordings_marathi/'+str(x)+ '_file_new.wav'
        f.write(request.data)
        f.close()

        imgs = datapreprocess_off(path)
        noFrames = imgs.shape[0]
                        #imgs = torch.from_numpy(imgs)
                        #prob = model(imgs)
                        #prob = prob.detach().numpy()
        prob = []
        for x in imgs:
                x = x.reshape(1,1,129,501)
                prob.append(sessOn.run([labName],{inName: x.astype(np.float32)})[0])
        prob = np.exp(np.asarray(prob))
        prob = prob**2
        prob = prob.tolist()
        list_predicts = []
        print(prob)
        pre_list = [0, 0, 0, 0]
        langlist = []
        print(noFrames)
        for j in range(4):
                pre_list[j] = prob[0]
        for j in range(noFrames):
                pre_list[j%4]= prob[j]
                ans = np.array(pre_list[0])
                for u in range(1,4):
                        ans += np.array(pre_list[u])
                list_predicts.append(ans.tolist())
        print(list_predicts)
        ind = ""
        
        langlist = np.argmax(list_predicts,-1).tolist()
	
        lmax = max(langlist, key=langlist.count)
        for ini in langlist:
                ind = ind + str(ini) + ' '
        
        
        return str(lmax)




i = 0


@app.route('/messages', methods = ['POST'])
def api_message():
        global i
        print("trial 1")
        f = open('./recordings/'+str(i)+ 'file_new.wav', 'wb')
        print("trial 2")
        
        
        path = './recordings/'+str(i)+ 'file_new.wav'    
        i = i + 1
        f.write(request.data)
        f.close()
        print(type(request.data))
        
        #print(fs)
        #print(len(audio))
        
        
        imgs = dataprepocess(path)
        noFrames = imgs.shape[0]
        #imgs = torch.from_numpy(imgs)
        #prob = model(imgs)
        #prob = prob.detach().numpy()
        prob = []
        for x in imgs:
            x = x.reshape(1,1,129,501)
            prob.append(sessOn.run([labName],{inName: x.astype(np.float32)})[0])
        
        prob = np.exp(np.asarray(prob))
        prob = prob**2
        prob = prob.tolist()
        list_predicts = []
        print(prob)
        pre_list = [0, 0, 0, 0]
        langlist = []
        print(noFrames)
        for i in range(4):
                pre_list[i] = prob[0]
        for i in range(noFrames):
                pre_list[i%4]= prob[i]
                ans = np.array(pre_list[0])
                for u in range(1,4):
                        ans += np.array(pre_list[u])
                list_predicts.append(ans.tolist())
        print(list_predicts)
        langlist = np.argmax(list_predicts,-1).tolist()

        listy = ""
        for el in langlist:
            listy = listy + str(el)

        return listy
        
    
        

@app.route('/', methods=['POST'])
def upload_file():
        if request.method == 'POST':
                # check if the post request has the file part
                link = request.form["linku"]
                if 'file' not in request.files:
                        flash('No file part')
                        return redirect(request.url)
                file = request.files['file']
                if file.filename == '':
                        flash('No file selected for uploading')
                        return redirect(request.url)
                if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                        nn = 'templates/'+filename
                        nnn = 'home/aih18/AIH19T-0161/UI/'+nn
                        # Audio -> Image -> Probability
                        imgs = datapreprocess_off(nn)
                        noFrames = imgs.shape[0]
                        #imgs = torch.from_numpy(imgs)
                        #prob = model(imgs)
                        #prob = prob.detach().numpy()
                        prob = []
                        for x in imgs:
                            x = x.reshape(1,1,129,501)
                            prob.append(sess.run([label_name],{input_name: x.astype(np.float32)})[0])
                        prob = np.exp(np.asarray(prob))
                        prob = prob**2
                        prob = prob.tolist()
                        list_predicts = []
                        print(prob)
                        pre_list = [0, 0, 0, 0]
                        langlist = []
                        print(noFrames)
                        for i in range(4):
                                pre_list[i] = prob[0]
                        for i in range(noFrames):
                                pre_list[i%4]= prob[i]
                                ans = np.array(pre_list[0])
                                for u in range(1,4):
                                        ans += np.array(pre_list[u])
                                list_predicts.append(ans.tolist())
                        print(list_predicts)
                        langlist = np.argmax(list_predicts,-1).tolist()
                        '''
                        if noFrames == 1:
                                ans = prob
                        else:
                                ans = np.array(prob[0])
                                for i in range(1,noFrames):
                                        ans = np.multiply(ans,np.array(prob[i]))
                                ans = list(ans)
                        probabilites = [float(i)/sum(ans) for i in ans]
                        #probabilites = [i*i for i in range(5)]
            # probabilites is the required ouput (List of 5 prob)

                        top_lang = Nmaxelements(probabilites, 3)
                        top_lang_str = ""
                        top_lang_str = lang_list[top_lang[0]]
                        flash(top_lang_str)

                        top_lang_str = lang_list[top_lang[1]]
                        flash(top_lang_str)

                        top_lang_str = lang_list[top_lang[2]]

                        flash(top_lang_str)
                        lolilist = [0,0,1,1,3,4,4,4,4,4,2,2,2]
                        '''
                        print(langlist)
                        return render_template('upload.html',loaded = 1, answers = langlist, apath = nnn, newlink = link)
        else:
                flash('Only \'.wav\' and \'.mp3\' files are allowed.')
                return redirect(request.url)




if __name__ == "__main__":
#       app.debug = True
 #       serveserve(app, host='0.0.0.0', port=9005)
        app.run(host='0.0.0.0', port=9005, ssl_context=('mycert.pem', 'mykey.pem'))


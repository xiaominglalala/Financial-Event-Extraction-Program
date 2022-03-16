#set mark_record.txt to be the id you start with. 1 means start form the first one.
#your marks are stored as manual_mark.txt
#fine_event.py will generate the raw_event_sent.txt as the raw mark input
import tkinter as tk
import pandas as pd

doc_id = 0
sent_id = 0
id = 1
source, raw, target, last1, last2, cur = 0,0,0,0,0,0
last_t1,last_t2 = 0,0
#left=1 意为当前不可回看
left = 1

def set_source():
    source = "qihuowang_content_with_keywords.csv"
    raw_mark = "raw_event_sent.txt"
    target = "manually_mark.txt"

    s = pd.read_csv(source)
    r = open(raw_mark,'r')
    t = open(target, 'a')
    last_t1 = last_t2 = t.tell()

    return s,r,t

def save():
    f = open('mark_record.txt','w')
    f.write(str(id))
    f.close()
    global target,raw

    target.close()
    raw.close()
    window.destroy()

#move source/raw_mark stream to the right place, return current raw mark
def jump(s,r):
    global doc_id, sent_id,last1,last2,id
    try:
        f = open('mark_record.txt','r')
        tmp = int(f.readline())
        id = tmp
        while tmp>1:
            discard = r.readline()
            tmp -= 1

            if not discard: #reach the end
                return []

        if tmp==1:
            last1 = last2 = r.tell()
            cur = r.readline()
            info = [int(c) for c in cur.split(' ')]
            if len(info)<10:
                return []
            doc_id = info[0]-1
            sent_id = info[1]
            #print(info)
            return info
    except:
        return []

def processKeyboard(ke):
    # 处理鼠标事件，ke为控件传递过来的键盘事件对象
    global id,doc_id,sent_id,source,raw,last1,last2,last_t1,last_t2,target,left
    if ke.keysym == 'Left':
        if left==1:
            return
        left = 1
        id -= 1
        if id ==0:
            id = 1
        #move cursor back
        raw.seek(last2)
        target.seek(last_t2)
        last_t1 = last_t2

        # 清空当前文本框内容
        text.delete(1.0, tk.END)

        cur = raw.readline()
        info = [int(c) for c in cur.split(' ')]
        if len(info) < 10:
            return
        doc_id = info[0] - 1
        sent_id = info[1]

        doc = source.loc[doc_id, 'mainpart']
        sent = doc[info[2]:info[3]]
        casue_b = info[4]
        cause_e = info[5]
        effect_b = info[6]
        effect_e = info[7]
        trigger = info[8:]
        # print(trigger)

        # "insert" 索引表示插入光标当前的位置
        for i in range(len(sent)):
            if i >= casue_b and i < cause_e:
                text.insert('end', sent[i] + '(' + str(i) + ')', 'cause')
            elif i >= effect_b and i < effect_e:
                text.insert('end', sent[i] + '(' + str(i) + ')', 'effect')
            else:
                flag = 0
                for j in range(0, len(trigger), 2):
                    if i >= trigger[j] and i < trigger[j + 1]:
                        flag = 1
                        text.insert('end', sent[i] + '(' + str(i) + ')', 'trigger')
                        break
                if flag == 0:
                    text.insert('end', sent[i] + '(' + str(i) + ')')

    elif ke.keysym =='Return':

        #清空当前文本框内容
        text.delete(1.0, tk.END)
        left = 0

        #转到下一文本
        id += 1
        print(id)
        last2 = last1
        last1 = raw.tell()
        #下面的部分和left里面相同
        cur = raw.readline()
        info = [int(c) for c in cur.split(' ')]
        if len(info)<10:
            return
        doc_id = info[0]-1
        sent_id = info[1]

        doc = source.loc[doc_id, 'mainpart']
        ###change to the spliter
        sent = doc[info[2]:info[3]]
        casue_b = info[4]
        cause_e = info[5]
        effect_b = info[6]
        effect_e = info[7]
        trigger = info[8:]
        # print(trigger)


        # "insert" 索引表示插入光标当前的位置
        for i in range(len(sent)):
            if i >= casue_b and i < cause_e:
                text.insert('end', sent[i] + '(' + str(i) + ')', 'cause')
            elif i >= effect_b and i < effect_e:
                text.insert('end', sent[i] + '(' + str(i) + ')', 'effect')
            else:
                flag = 0
                for j in range(0, len(trigger), 2):
                    if i >= trigger[j] and i < trigger[j + 1]:
                        flag = 1
                        text.insert('end', sent[i] + '(' + str(i) + ')', 'trigger')
                        break
                if flag == 0:
                    text.insert('end', sent[i] + '(' + str(i) + ')')

def get_mark(e):
    global mark,E1,target, last_t1,last_t2
    manual = mark.get()
    #print(manual)
    E1.delete(0, tk.END)

    last_t2 = last_t1
    last_t1 = target.tell()
    #print(last_t1,last_t2)
    target.write(str(id)+' '+str(doc_id+1)+' '+str(sent_id)+' '+manual+'\n')


# 窗口和标题
window = tk.Tk()
window.title("mark_tool")
# 绑定鼠标键盘事件，交由processKeyboardEvent函数去处理，事件对象会作为参数传递给该函数
window.bind("<KeyPress>", processKeyboard)
# 设置退出窗口自动保存
window.protocol('WM_DELETE_WINDOW', save)

text = tk.Text(window, width=100, height=20)
text.pack()

# 设置 tag
text.tag_config("cause", foreground="red")
text.tag_config("effect", foreground="blue")
text.tag_config("trigger", background="yellow")
text.tag_config("idex",foreground = "violet")

#设置输入框
L1 = tk.Label(window, text="因，果，关键词，可输入多段，如：0-2 3-4 5-6/8-9")
L1.pack(side = tk.LEFT)
mark = tk.StringVar()
E1 = tk.Entry(window, bd =5, textvariable = mark, width=50)
E1.pack(side = tk.RIGHT)
E1.bind('<Return>', get_mark)




source, raw, target = set_source()
cur = jump(source, raw)

if len(cur)<10:
    print("mark_record error!")
else:
    doc = source.loc[int(doc_id),'mainpart']
    ###change to the spliter

    sent = doc[cur[2]:cur[3]]
    #print(doc_id,sent_id)
    casue_b = cur[4]
    cause_e = cur[5]
    effect_b = cur[6]
    effect_e = cur[7]
    trigger = cur[8:]
    #print(trigger)


    # "insert" 索引表示插入光标当前的位置
    for i in range(len(sent)):
        if i>=casue_b and i<cause_e:
            text.insert('end',sent[i]+'('+str(i)+')','cause')
        elif i>=effect_b and i<effect_e:
            text.insert('end', sent[i]+'('+str(i)+')', 'effect')
        else:
            flag = 0
            for j in range(0,len(trigger),2):
                if i>=trigger[j] and i<trigger[j+1]:
                    flag = 1
                    text.insert('end',sent[i]+'('+str(i)+')', 'trigger')
                    break
            if flag == 0:
                text.insert('end', sent[i]+'('+str(i)+')')

    # 消息循环
    window.mainloop()




#text.delete(1.0,Tkinter.END)



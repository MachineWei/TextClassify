# -*- coding: utf-8 -*-
'''
部署时代码
'''

import time
import torch
import jieba
from models import TextCNN, TextCNNConfig


class Classify:

    def __init__(self, features='word', device='gpu'):
        self.features = features
        self.sentence_length = TextCNNConfig.sequence_length
        self.device = device
        self.__device()
        self.load_vocab()
        self.__load_model()

    def __device(self):
        if torch.cuda.is_available() and self.device=='gpu':
            self.device = torch.device('cuda')
        else:
            self.device = 'cpu'

    def __load_model(self):
        self.model = TextCNN(TextCNNConfig)
        self.model.load_state_dict(torch.load("./ckpts/cnn_model.pth"))
        self.model.to(self.device)
        self.model.eval()

    def load_vocab(self):
        with open('./ckpts/vocab.txt','r',encoding='utf-8') as f:
            vocab = f.read().strip().split('\n')
        self.vocab = {k: v for k, v in zip(vocab, range(len(vocab)))}

        with open('./ckpts/target.txt','r',encoding='utf-8') as f:
            target = f.read().strip().split('\n')
        self.target = {v: k for k, v in zip(target, range(len(target)))}        

    def cut_words(self, sentence : str) -> list:
        if self.features == 'word':
            return jieba.lcut(sentence)
        else:
            return list(sentence)

    def sentence_cut(self, sentence):
        """针对一个句子的字符转ID，并截取到固定长度，返回定长的字符代号。"""
        words = self.cut_words(sentence)
        if len(words) >= self.sentence_length:
            sentence_cutted = words[:self.sentence_length]
        else:
            sentence_cutted = words + ["<PAD>"] * (self.sentence_length - len(words))
        sentence_id = [self.vocab[w] if w in self.vocab else self.vocab["<UNK>"] for w in sentence_cutted]
        return sentence_id

    def predict(self, content):
        """
        传入一个句子，测试单个类别
        """
        with torch.no_grad():
            content_id = [self.sentence_cut(content)]
            start_time = time.time()
            content_id = torch.LongTensor(content_id)
            one_batch_input = content_id.to(self.device)
            outputs = self.model(one_batch_input)
            max_value, max_index = torch.max(outputs, axis=1)
            predict = max_index.cpu().numpy()
            print(time.time()-start_time)
        return self.target[predict[0]]


if __name__ == '__main__':
    classify = Classify()
    text = '鲍勃库西奖归谁属？ NCAA最强控卫是坎巴还是弗神新浪体育讯如今，本赛季的NCAA进入到了末段，各项奖项的评选结果也即将出炉，其中评选最佳控卫的鲍勃-库西奖就将在下周最终四强战时公布，鲍勃-库西奖是由奈史密斯篮球名人堂提供，旨在奖励年度最佳大学控卫。最终获奖的球员也即将在以下几名热门人选中产生。〈〈〈 NCAA疯狂三月专题主页上线，点击链接查看精彩内容吉梅尔-弗雷戴特，杨百翰大学“弗神”吉梅尔-弗雷戴特一直都备受关注，他不仅仅是一名射手，他会用“终结对手脚踝”一样的变向过掉面前的防守者，并且他可以用任意一支手完成得分，如果他被犯规了，可以提前把这两份划入他的帐下了，因为他是一名命中率高达90%的罚球手。弗雷戴特具有所有伟大控卫都具备的一点特质，他是一位赢家也是一位领导者。“他整个赛季至始至终的稳定领导着球队前进，这是无可比拟的。”杨百翰大学主教练戴夫-罗斯称赞道，“他的得分能力毋庸置疑，但是我认为他带领球队获胜的能力才是他最重要的控卫职责。我们在主场之外的比赛(客场或中立场)共取胜19场，他都表现的很棒。”弗雷戴特能否在NBA取得成功？当然，但是有很多专业人士比我们更有资格去做出这样的判断。“我喜爱他。”凯尔特人主教练多克-里弗斯说道，“他很棒，我看过ESPN的片段剪辑，从剪辑来看，他是个超级巨星，我认为他很成为一名优秀的NBA球员。”诺兰-史密斯，杜克大学当赛季初，球队宣布大一天才控卫凯瑞-厄尔文因脚趾的伤病缺席赛季大部分比赛后，诺兰-史密斯便开始接管球权，他在进攻端上足发条，在ACC联盟(杜克大学所在分区)的得分榜上名列前茅，但同时他在分区助攻榜上也占据头名，这在众强林立的ACC联盟前无古人。“我不认为全美有其他的球员能在凯瑞-厄尔文受伤后，如此好的接管球队，并且之前毫无准备。”杜克主教练迈克-沙舍夫斯基赞扬道，“他会将比赛带入自己的节奏，得分，组织，领导球队，无所不能。而且他现在是攻防俱佳，对持球人的防守很有提高。总之他拥有了辉煌的赛季。”坎巴-沃克，康涅狄格大学坎巴-沃克带领康涅狄格在赛季初的毛伊岛邀请赛一路力克密歇根州大和肯塔基等队夺冠，他场均30分4助攻得到最佳球员。在大东赛区锦标赛和全国锦标赛中，他场均27.1分，6.1个篮板，5.1次助攻，依旧如此给力。他以疯狂的表现开始这个赛季，也将以疯狂的表现结束这个赛季。“我们在全国锦标赛中前进着，并且之前曾经5天连赢5场，赢得了大东赛区锦标赛的冠军，这些都归功于坎巴-沃克。”康涅狄格大学主教练吉姆-卡洪称赞道，“他是一名纯正的控卫而且能为我们得分，他有过单场42分，有过单场17助攻，也有过单场15篮板。这些都是一名6英尺175镑的球员所完成的啊！我们有很多好球员，但他才是最好的领导者，为球队所做的贡献也是最大。”乔丹-泰勒，威斯康辛大学全美没有一个持球者能像乔丹-泰勒一样很少失误，他4.26的助攻失误在全美遥遥领先，在大十赛区的比赛中，他平均35.8分钟才会有一次失误。他还是名很出色的得分手，全场砍下39分击败印第安纳大学的比赛就是最好的证明，其中下半场他曾经连拿18分。“那个夜晚他证明自己值得首轮顺位。”当时的见证者印第安纳大学主教练汤姆-克雷恩说道。“对一名控卫的所有要求不过是领导球队、使球队变的更好、带领球队成功，乔丹-泰勒全做到了。”威斯康辛教练博-莱恩说道。诺里斯-科尔，克利夫兰州大诺里斯-科尔的草根传奇正在上演，默默无闻的他被克利夫兰州大招募后便开始刻苦地训练，去年夏天他曾加练上千次跳投，来提高这个可能的弱点。他在本赛季与杨斯顿州大的比赛中得到40分20篮板和9次助攻，在他之前，过去15年只有一位球员曾经在NCAA一级联盟做到过40+20，他的名字是布雷克-格里芬。“他可以很轻松地防下对方王牌。”克利夫兰州大主教练加里-沃特斯如此称赞自己的弟子，“同时他还能得分，并为球队助攻，他几乎能做到一个成功的团队所有需要的事。”这其中四名球员都带领自己的球队进入到了甜蜜16强，虽然有3个球员和他们各自的球队被挡在8强的大门之外，但是他们已经表现的足够出色，不远的将来他们很可能出现在一所你熟悉的NBA球馆里。(clay)'
    result = classify.predict(text)
    print(result)







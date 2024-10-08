import json
from pycocotools.coco import COCO
from collections import defaultdict
from nltk.tokenize import word_tokenize

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# COCOEvalCap class
class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

        self.gts = None
        self.res = None

    def tokenize(self):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        print('Tokenizing captions...')
        tokenizer = PTBTokenizer()
        self.gts = tokenizer.tokenize(gts)
        self.res = tokenizer.tokenize(res)

    def evaluate(self):
        self.tokenize()

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

        for scorer, method in scorers:
            print(f'Computing {scorer.method()} score...')
            score, scores = scorer.compute_score(self.gts, self.res)
            if isinstance(method, list):
                for idx, m in enumerate(method):
                    sc = score[idx]
                    scs = scores[idx]
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, self.gts.keys(), m)
                    print(f"{m}: {sc:.3f}")
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, self.gts.keys(), method)
                print(f"{method}: {score:.3f}")
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        if isinstance(scores, dict):
            for imgId in imgIds:
                if imgId not in self.imgToEval:
                    self.imgToEval[imgId] = {}
                    self.imgToEval[imgId]["image_id"] = imgId
                self.imgToEval[imgId][method] = scores[imgId]
        elif isinstance(scores, list):
            for imgId, score in zip(imgIds, scores):
                if imgId not in self.imgToEval:
                    self.imgToEval[imgId] = {}
                    self.imgToEval[imgId]["image_id"] = imgId
                self.imgToEval[imgId][method] = score
        else:
            for imgId in imgIds:
                if imgId not in self.imgToEval:
                    self.imgToEval[imgId] = {}
                    self.imgToEval[imgId]["image_id"] = imgId
                self.imgToEval[imgId][method] = scores

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

def evaluate_captions(ground_truth_file, generated_file):
    # Load ground truth and generated captions
    coco = COCO(ground_truth_file)

    # Load generated captions
    cocoRes = COCO()
    with open(generated_file, 'r') as f:
        generated_captions = json.load(f)
        cocoRes.dataset = {'annotations': generated_captions['annotations']}
        cocoRes.createIndex()

    # Create COCOEvalCap object
    cocoEval = COCOEvalCap(coco, cocoRes)

    # Evaluate
    cocoEval.evaluate()

    # Print evaluation results
    print(f"Evaluation results for {generated_file}:")
    for metric, score in cocoEval.eval.items():
        print(f'{metric}: {score:.3f}')

# Evaluate captions for the first and second set
evaluate_captions('ground_truth_captions1.json', 'generated_captions1.json')
evaluate_captions('ground_truth_captions2.json', 'generated_captions2.json')

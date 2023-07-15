from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                total_scores["Bleu"] = score
            else:
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))

        return total_scores


# scorer = Scorer(ref, gt)
# scorer.compute_scores()

# HW 3
# Cumhur Aygar
import collections
import math
import argparse


class IRSystem:

    def __init__(self, f):
        # Data structures to store documents and term statistics
        self.docs = {}
        self.df = collections.Counter()
        self.N = 0
        self.doc_vectors = {}
        self.doc_norms = {}

        # Read each line of the corpus
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc_id_str, text = line.split(maxsplit=1)
            doc_id = int(doc_id_str)

            terms = text.lower().split()
            tf = collections.Counter(terms)
            self.docs[doc_id] = tf

            for term in set(tf):
                self.df[term] += 1

            self.N += 1

        for doc_id, tf_counts in self.docs.items():
            vec = {}
            for term, freq in tf_counts.items():
                weight = 1 + math.log10(freq)
                vec[term] = weight

            norm = math.sqrt(sum(w ** 2 for w in vec.values()))
            if norm > 0:
                for term in vec:
                    vec[term] /= norm

            self.doc_vectors[doc_id] = vec
            self.doc_norms[doc_id] = norm

    def run_query(self, query):
        terms = query.lower().split()
        return self._run_query(terms)

    def _run_query(self, terms):
        # ltn weighting for query:
        #   l: 1 + log10(freq)
        #   t: log10(N / df)
        #   n: no normalization
        q_tf = collections.Counter(terms)
        query_vec = {}

        for term, freq in q_tf.items():
            if freq > 0 and term in self.df:
                tf_l = 1 + math.log10(freq)
                idf = math.log10(self.N / self.df[term])
                query_vec[term] = tf_l * idf

        scores = {}
        for doc_id, d_vec in self.doc_vectors.items():
            score = 0.0
            for term, q_weight in query_vec.items():
                if term in d_vec:
                    score += q_weight * d_vec[term]
            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if not ranked or ranked[0][1] == 0:
            return list(range(10))

        return [doc_id for doc_id, _ in ranked[:10]]


def main(corpus):
    ir = IRSystem(open(corpus))
    while True:
        query = input('Query: ').strip()
        if query.lower() == 'exit':
            break
        results = ir.run_query(query)
        print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("CORPUS", help="Path to file with the corpus")
    args = parser.parse_args()
    main(args.CORPUS)

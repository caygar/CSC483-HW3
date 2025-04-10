# DO NOT modify code except "YOUR CODE GOES HERE" blocks

from pytest import approx

import tfidf_engine

ir = tfidf_engine.IRSystem(open("wiki-small.txt"))


def test_1word_query_10points():
    results = ir.run_query('house')
    print(results)
    print([1617, 1096, 5329, 3765, 4612, 2782, 1757, 1581, 1820, 4465])
    assert results == [1617, 1096, 5329, 3765, 4612, 2782, 1757, 1581, 1820, 4465]


def test_1word_repeated1_query_10points():
    results = ir.run_query('house house')
    print(results)
    print([1617, 1096, 5329, 3765, 4612, 2782, 1757, 1581, 1820, 4465])
    assert results == [1617, 1096, 5329, 3765, 4612, 2782, 1757, 1581, 1820, 4465]


def test_1word_repeated2_query_10points():
    results = ir.run_query('house house house')
    print(results)
    print([1617, 1096, 5329, 3765, 4612, 2782, 1757, 1581, 1820, 4465])
    assert results == [1617, 1096, 5329, 3765, 4612, 2782, 1757, 1581, 1820, 4465]


def test_1word_unknown_query_10points():
    results = ir.run_query('aha')
    assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_1word_infrequent_query_10points():
    results = ir.run_query('eduardo')
    assert results == [5199, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_2words_query_10points():
    results = ir.run_query('house white')
    assert results == [5115, 273, 5486, 4511, 5077, 1546, 576, 1617, 4759, 577]


def test_2words_reversed_query_10points():
    results = ir.run_query('white house')
    assert results == [5115, 273, 5486, 4511, 5077, 1546, 576, 1617, 4759, 577]


def test_1word_infrequent_query1_10points():
    results = ir.run_query('racehorse')
    assert results == [776, 2667, 3873, 5130, 0, 1, 2, 3, 4, 5]


def test_1word_infrequent_query2_10points():
    results = ir.run_query('regulations')
    assert results == [955, 3050, 5746, 37, 1256, 0, 1, 2, 3, 4]


def test_2words_infrequent_query_10points():
    results = ir.run_query('racehorse regulations')
    assert results == [776, 2667, 3873, 5130, 955, 3050, 5746, 37, 1256, 0]


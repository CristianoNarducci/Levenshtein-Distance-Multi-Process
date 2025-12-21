import string
import time
import concurrent.futures
import os
import random
import multiprocessing

def random_string(length: int):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_dataset(n_string,length):
    return [random_string(length) for _ in range(n_string)]

def chunkify(data, n_chunks):
    size = (len(data)+ n_chunks - 1)
    return [data[i:i + size] for i in range(0,len(data),size)]

def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[m][n]

def worker(args):
    pattern,chunk = args
    return [levenshtein_distance(pattern,s) for s in chunk]

def main():
    n_proc = multiprocessing.cpu_count()
    print(n_proc)
    pattern = random_string(100)
    corpus = generate_dataset(50000,120)
    chunks = chunkify(corpus,n_proc)
    args = [(pattern, chunk) for chunk in chunks]
    print(f"pattern: {pattern}\n")
    print("Start MultiProcess Version:")
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers= n_proc) as executor:
        results = list(executor.map(worker,args))
    end = time.time()
    results = [d for sub in results for d in sub]
    print(f"multiprocessing time: {end - start}\n")
    print("Start Sequential Version:")
    results.clear()
    start = time.time()
    for string in corpus:
        results.append(levenshtein_distance(pattern,string))
    end = time.time()
    print(f"sequential time: {end - start}")

if __name__ == '__main__':
    main()
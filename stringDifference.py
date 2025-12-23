import string
import time
import concurrent.futures
import random
import os
import multiprocessing


def load_corpus(path):
    corpus = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            parts = line.split(']')
            text = line[-1].strip() if len(parts) > 1 else line.strip()

            if text:
                corpus.append(text)
    return corpus

def random_string(length: int):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def generate_dataset(n_string, length):
    return [random_string(length) for _ in range(n_string)]


def chunkify(data, n_chunks):
    size = max(1, len(data) // n_chunks)
    return [data[i:i + size] for i in range(0, len(data), size)]


def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]


def bitap(s1: str, s2: str, k: int):
    m = len(s2)
    if m == 0:
        return []
    alphabet = set(s1 + s2)
    mask = {}
    for i, c in enumerate(s2):
        mask[c] = mask.get(c, ~0) & ~(1 << i)

    R = [~0] * (k + 1)
    matches = []

    for i, c in enumerate(s1):
        old_R = list(R)
        R[0] = ((old_R[0] << 1) | 1) | mask.get(c, ~0)
        for j in range(1, k + 1):
            # Bitap con approssimazione (Levenshtein/Hamming semplificato)
            R[j] = ((old_R[j] << 1) | 1) | mask.get(c, ~0)
            R[j] &= (old_R[j - 1] << 1) & (R[j - 1] << 1) & old_R[j - 1]
        if not (R[k] & (1 << (m - 1))):
            matches.append(i)

    return matches


def worker_bitap(args):
    pattern, chunk, index = args
    k = 1
    print(f"process id: {os.getpid()} process chunk {index}")
    return [bitap(s, pattern, k) for s in chunk]


def worker_levenshtein(args):
    pattern, chunk, index = args
    print(f"process id: {os.getpid()} process chunk {index}")
    return [(word, levenshtein_distance(pattern.lower(), word)) for word in chunk]

def main():
    n_proc = multiprocessing.cpu_count()
    print(f"Core rilevati: {n_proc}")
    user_choice = int(input("Use random string generator(1) or sample dataset (2)? "))
    if user_choice == 1:
        pattern = random_string(100)#"product"
        corpus = generate_dataset(50000, 120)
    elif user_choice == 2:
        pattern = str(input("input a word ")).strip()
        corpus = load_corpus('dataset.txt')
    else:
        print("error wrong input!")
        return
    print(f"Dataset caricato: {len(corpus)} elementi.")
    all_text = " ".join(corpus).translate(str.maketrans('', '', string.punctuation)).lower()
    unique_words = list(set(all_text.split()))
    print(f"Analisi su {len(unique_words)} parole uniche trovate nel dataset.")
    chunks_lev = chunkify(unique_words, n_proc)
    args = [(pattern, chunk, i) for i, chunk in enumerate(chunks_lev)]
    print(f"pattern: {pattern}\n")
    print("Start MultiProcess Version:")
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
        results = list(executor.map(worker_levenshtein, args))
    end = time.perf_counter()
    time_parallel = end - start
    print(f"\nMultiprocess time elapsed: {time_parallel:.3f}s")
    final_results = [item for sublist in results for item in sublist]
    # Ordiniamo per distanza (i più simili in alto)
    final_results.sort(key=lambda x: x[1])
    print("--- Top 10 parole più simili nel dataset ---")
    for word, dist in final_results[:10]:
        print(f"Parola: {word:15} | Distanza: {dist}")
    print("\nStart Sequential Version:")
    results = []
    start = time.perf_counter()
    for words in unique_words:
        results.append(levenshtein_distance(pattern.lower(), words.lower()))
    end = time.perf_counter()
    time_sequential = end - start
    print(f"Sequential time elapsed: {time_sequential:.3f} s,with index: {results.index(max(results))}")
    print(f"Speedup: {time_sequential / time_parallel:.3f}")
    print("\nStart Bitap algorithm parallelized")
    chunks_bit = chunkify(corpus,n_proc)
    args = [(pattern, chunk, i) for i, chunk in enumerate(chunks_bit)]
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
        bitap_raw = list(executor.map(worker_bitap, args))
    end = time.perf_counter()
    time_parallel = end - start
    bitap_results = [res for sub in bitap_raw for res in sub]
    matches_found = [i for i, m in enumerate(bitap_results) if len(m) > 0]
    print(f"\nMultiprocess time elapsed: {time_parallel:.3f}s")
    print(f"Stringhe con match: {len(matches_found)} su {len(corpus)}")
    if matches_found:
        idx = matches_found[0]
        print(f"Esempio: Stringa {idx} ha match alle posizioni: {bitap_results[idx]} con testo: {corpus[idx]}")
    print("\nStart Sequential Version:")
    start = time.perf_counter()
    results_seq = []
    for text in corpus:
        results_seq.append(bitap(text, pattern, k=1))
    end = time.perf_counter()
    time_sequential = end - start
    print(f"sequential time elapsed: {time_sequential:.3f}")
    print(f"Speedup:{time_sequential/time_parallel:.3f}")
    matches_found = [i for i, m in enumerate(results_seq) if len(m) > 0]
    print(f"Stringhe con match: {len(matches_found)} su {len(corpus)}")
    if matches_found:
        idx = matches_found[0]
        print(f"Esempio: Stringa {idx} ha match alle posizioni: {bitap_results[idx]} con testo: {corpus[idx]}")


if __name__ == '__main__':
    main()
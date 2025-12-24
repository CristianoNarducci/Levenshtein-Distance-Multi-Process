import string
import time
import concurrent.futures
import random
import multiprocessing
from tqdm import tqdm

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
    for i in range(0,len(data),size):
        yield data[i:i + size]


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


def bitap(text):
    R = [~0] * (GLOBAL_K + 1)
    matches = []

    for i, c in enumerate(text):
        char_mask = GLOBAL_MASK.get(c, ~0)

        old_R0 = R[0]
        R[0] = ((old_R0 << 1) | 1) | char_mask

        for j in range(1, GLOBAL_K + 1):
            tmp = R[j]
            R[j] = ((tmp << 1) | 1) | char_mask
            R[j] &= (old_R0 << 1) & (R[j - 1] << 1) & old_R0
            old_R0 = tmp

        if not (R[GLOBAL_K] & (1 << (GLOBAL_M - 1))):
            matches.append(i)

    return matches


def init_worker_lev(pattern):
    global GLOBAL_PATTERN
    GLOBAL_PATTERN = pattern.lower()

def init_worker_bitap(pattern,k):
    global GLOBAL_PATTERN, GLOBAL_MASK, GLOBAL_K, GLOBAL_M
    GLOBAL_PATTERN = pattern.lower()
    GLOBAL_K = k
    GLOBAL_M = len(pattern.lower())
    mask = {}
    for i, c in enumerate(pattern):
        mask[c] = mask.get(c, ~0) & ~(1 << i)
    GLOBAL_MASK = mask

def worker_bitap(chunk):
    result = []
    for text in chunk:
        result.append(bitap(text))
    return result

def worker_levenshtein(args):
    chunk = args
    return [(word, levenshtein_distance(GLOBAL_PATTERN, word)) for word in chunk]

def main():
    n_proc = max(1, multiprocessing.cpu_count() - 2)
    print(f"Core totali: {multiprocessing.cpu_count()} | Core utilizzati: {n_proc}")
    num_chunks = n_proc * 10
    user_choice = int(input("Use random string generator (1) or sample dataset (2)? "))
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
    unique_words = set()
    for text in corpus:
        clean_text = text.translate(str.maketrans('','',string.punctuation)).lower()
        unique_words.update(clean_text.split())
    unique_words = list(unique_words)
    print(f"Analisi su {len(unique_words)} parole uniche trovate nel dataset.")
    chunks_lev = chunkify(unique_words, num_chunks)
    args = [chunk for _, chunk in enumerate(chunks_lev)]
    print(f"Avvio elaborazione con {n_proc} processi su {len(args)} chunk...\n")
    print(f"pattern: {pattern}\n")
    print("Start MultiProcess Version:")
    final_results = []
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc, initializer=init_worker_lev, initargs=(pattern,)) as executor:
        future_to_chunk = {executor.submit(worker_levenshtein, arg): arg for arg in args}
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk),total=len(future_to_chunk),desc="elaborazione chunk",unit=" chunk"):
            try:
                result = future.result()
                final_results.extend(result)
            except Exception as exc:
                print(f"Un chunk ha generato un errore: {exc}")
    end = time.perf_counter()
    time_parallel = end - start
    print(f"\nMultiprocess time elapsed: {time_parallel:.3f}s")
    final_results.sort(key=lambda x: x[1])
    print("--- Top 10 parole più simili nel dataset ---")
    for word, dist in final_results[:10]:
        print(f"Parola: {word:15} | Distanza: {dist}")

    print("\nStart Sequential Version:")
    results = []
    start = time.perf_counter()
    for words in tqdm(unique_words,total=len(unique_words),desc="elaborazione chunk",unit=" chunk"):
        results.append(levenshtein_distance(pattern.lower(), words.lower()))
    end = time.perf_counter()
    time_sequential = end - start
    print(f"Sequential time elapsed: {time_sequential:.3f} s,with index: {results.index(max(results))}")
    print(f"Speedup: {time_sequential / time_parallel:.3f}")
    print("\nStart Bitap algorithm parallelized")
    chunks_bit = chunkify(corpus,num_chunks)
    args = [chunk for _, chunk in enumerate(chunks_bit)]
    bitap_raw = []
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc,initializer=init_worker_bitap,initargs=(pattern,1)) as executor:
        futures = {executor.submit(worker_bitap, arg): arg for arg in args}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Bitap", unit=" chunk"):
            bitap_raw.extend(future.result())
    end = time.perf_counter()
    time_parallel = end - start
    matches_found = [i for i, m in enumerate(bitap_raw) if len(m) > 0]
    print(f"\nMultiprocess time elapsed: {time_parallel:.3f}s")
    print(f"Stringhe con match: {len(matches_found)} su {len(corpus)}")
    if matches_found:
        idx = matches_found[0]
        print(f"Esempio: Stringa {idx}: {corpus[idx]}")
    print("\nStart Sequential Version:")
    init_worker_bitap(pattern, 1)
    start = time.perf_counter()
    results_seq = []
    for text in tqdm(corpus,total=len(corpus),desc="elaborazione chunk",unit=" chunk"):
        results_seq.append(bitap(text))
    end = time.perf_counter()
    time_sequential = end - start
    print(f"sequential time elapsed: {time_sequential:.3f}")
    print(f"Speedup: {time_sequential/time_parallel:.3f}")
    matches_found = [i for i, m in enumerate(results_seq) if len(m) > 0]
    print(f"Stringhe con match: {len(matches_found)} su {len(corpus)}")
    if matches_found:
        idx = matches_found[0]
        print(f"Esempio: Stringa {idx} testo: {corpus[idx]}")


if __name__ == '__main__':
    main()
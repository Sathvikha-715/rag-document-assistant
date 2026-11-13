import time
from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.rag_chain import ask_question

# Put the PDF next to this file, or use a full path like r"C:\Users\...\file.pdf"
PDF_PATH = "Data_Analysis_Types_Process_Methods_Techniques_and.pdf"

REFUSAL = ["couldn't find", "could not find", "not in the document",
           "not mentioned", "no information", "don't know"]

# (question, keywords, min keywords needed, expected PDF pages)
TESTS = [
    ("What are the five major types of data analysis?",
     ["text", "statistical", "diagnostic", "predictive", "prescriptive"], 4, {2}),
    ("What is prescriptive analysis?", ["action", "decision"], 1, {2}),
    ("What question does diagnostic analysis answer?", ["why"], 1, {2}),
    ("What does predictive analysis show?", ["likely", "future"], 1, {2}),
    ("What is text analysis also called?", ["data mining"], 1, {2}),
    ("What are the two categories of statistical analysis?",
     ["descriptive", "inferential"], 2, {2}),
    ("What does descriptive analysis show for continuous data?",
     ["mean", "deviation"], 1, {2}),
    ("What are the steps of the data analysis process?",
     ["collection", "cleaning", "communication"], 2, {2, 3}),
    ("What are the two categories of data collection methods?",
     ["primary", "secondary"], 2, {3}),
    ("How is primary data collection divided?",
     ["quantitative", "qualitative"], 2, {3}),
    ("Why is data cleaning important?", ["error", "misleading"], 1, {3}),
    ("What are the two types of data cleaning?",
     ["code", "contingency"], 2, {3}),
    ("What are the steps leading up to web data analysis?",
     ["identify", "extract", "prepare", "integrate", "consume"], 4, {3}),
    ("Who created Python and when?", ["guido", "1980"], 1, {4, 5}),
    ("What is R programming?", ["statistical", "free"], 1, {4}),
    ("What does Tableau Public do?", ["dashboard", "free"], 1, {4}),
    ("Who created SAS and when?", ["1966"], 1, {5}),
    ("How much faster is Apache Spark than Hadoop?", ["100"], 1, {5}),
    ("What does MLlib provide in Spark?", ["regression", "clustering"], 1, {5}),
    ("What are the business benefits of data analytics?",
     ["efficiency", "cost", "decision"], 2, {5, 6}),
    # Added after manual testing
    ("Which tools does the paper list for data analysis?",
     ["r programming", "tableau", "python", "sas", "spark", "excel", "rapidminer"],
     6, {4, 5}),
    ("Summarize the whole document in 3 lines.",
     ["types", "process", "tools"], 2, None),
    # Off-topic: should be refused
    ("Who won the 2022 FIFA World Cup?", REFUSAL, 1, None),
    ("What is the salary of a data analyst in India?", REFUSAL, 1, None),
]

print("Building index...")
docs = load_pdf(PDF_PATH)
vs = create_vectorstore(split_documents(docs))

correct = cited = cited_total = 0
times = []

for i, (q, kws, need, pages) in enumerate(TESTS, 1):
    start = time.time()
    try:
        answer, retrieved = ask_question(vs, q)
    except Exception as e:
        print(f"{i:>2}. ERROR  {q}  ->  {e}")
        continue
    times.append(time.time() - start)

    text = answer.lower()
    hits = sum(1 for k in kws if k in text)
    ok = hits >= need
    correct += ok

    cite_note = ""
    if pages is not None:
        got = {d.metadata.get("page", 0) + 1 for d in retrieved}
        cited_total += 1
        if got & pages:
            cited += 1
            cite_note = f"pages {sorted(got)} OK"
        else:
            cite_note = f"pages {sorted(got)} expected {sorted(pages)}"

        print(f"{i:>2}. {'PASS' if ok else 'FAIL'}  {times[-1]:.1f}s  {q}  [{cite_note}]")
    if not ok:
        print("    ANSWER GOT:", answer[:400].replace("\n", " "))
    time.sleep(1)  # avoid rate limits

n = len(times)
print("\n--- RESULTS ---")
print(f"Answer accuracy : {correct}/{len(TESTS)}")
print(f"Citation hit    : {cited}/{cited_total}")
print(f"Avg response    : {sum(times)/max(n,1):.2f} s")
# Best Practices for Quantitative Sociology Literature Review

This document outlines a recommended workflow for conducting a **quantitative literature review** in sociology, including ways to integrate automated tools such as LLMs and entity extraction.

---

## 1. Define Scope and Research Questions
- Clearly define the research question(s) or hypotheses.
- Determine inclusion/exclusion criteria (e.g., time period, population, study design).
- Identify key variables and outcome measures.

---

## 2. Systematic Search & Organization
- Use databases such as Web of Science, Scopus, JSTOR.
- Construct Boolean search queries and controlled vocabularies.
- Export bibliographic info to a reference manager (Zotero, Mendeley, EndNote).
- Optional: Use LLMs to suggest additional keywords or synonyms.

---

## 3. Screening & Selection
- Screen titles and abstracts for relevance.
- Perform full-text screening according to inclusion criteria.
- Document the screening process using a PRISMA flow diagram.
- Optional: LLM-assisted pre-screening of abstracts.

---

## 4. Data Extraction & Coding
- Extract quantitative info: sample size, measures, effect sizes, methods.
- Use structured templates (Excel, CSV, or database) for coding study characteristics.
- Optional: LLM or NER pipelines to automate variable extraction.

---

## 5. Synthesis & Analysis
- Narrative synthesis: summarize trends, gaps, and methodological differences.
- Meta-analysis if compatible effect sizes are available.
- Optional: Use LLMs to summarize trends across multiple papers.
- Visualization of results: histograms, correlation matrices, time trends.

---

## 6. Critical Appraisal
- Assess study quality, biases, and methodological rigor.
- Note limitations and generalizability.
- Highlight research gaps.

---

## 7. Writing the Literature Review
- Suggested structure: Introduction → Methods → Synthesis → Discussion → Gaps.
- Use templates to maintain consistency across sections.
- Tables/figures are useful for quantitative patterns.
- LLM-assisted drafting: generate section summaries or paragraph drafts.

---

## 8. Iterative & Transparent Process
- Track every step: search queries, inclusion decisions, extracted data.
- Update review periodically with new studies.
- Transparency improves reproducibility and credibility.

---

## Optional Automation Ideas
1. Entity & relation extraction for variables and outcomes.
2. Trend analysis over time using automated summaries.
3. Gap identification through LLMs highlighting under-studied areas.
4. Summary aggregation for high-level overview across multiple papers.

---

**Tip:** Always validate automated outputs manually. LLMs and automated pipelines accelerate workflow but cannot fully replace human judgment.


# Classical Practice and Emerging Practice

## Traditional Systematic Literature Review
Core Philosophy: Quality over quantity, deep reading, manual curation
Typical Steps:

Define Research Question (PICO framework)

Population, Intervention, Comparison, Outcome
Very specific, narrow scope


Protocol Development

Pre-register search strategy
Define inclusion/exclusion criteria upfront
PRISMA guidelines


Systematic Search (50-300 papers typical final set)

2-4 databases (PubMed, Scopus, Web of Science)
Boolean search strings
Hand searching key journals
Reference chaining (snowballing)


Screening Process

Initial: 500-2000 papers from search
Title/abstract screening → ~200-500 papers
Full-text review → ~50-200 papers
Often 2+ independent reviewers for reliability


Quality Assessment

Manual quality/bias assessment
Risk of bias tools (Cochrane, GRADE)
Study design evaluation


Data Extraction

Manual coding into structured forms
Key findings, methods, sample sizes
Inter-rater reliability checks


Synthesis

Narrative synthesis
Meta-analysis (quantitative)
Thematic analysis


Reporting

PRISMA flow diagram
Quality assessment tables
6-18 months typical duration



Strengths:

Deep understanding of each paper
High quality control
Well-established methodology
Accepted by all journals

Limitations:

Labor intensive (200-1000 hours)
Limited scope (misses broader context)
Slow (outdated by publication)
Publication bias
Cannot handle rapid fields (AI/ML)


## Modern LLM-Assisted Large-Scale Review
Core Philosophy: Comprehensive coverage, computational analysis, pattern discovery
Typical Steps:

Broad Question Definition

Can handle exploratory questions
Multiple research questions simultaneously
Adaptive scope refinement


Mass Collection (5,000-50,000 papers)

Multi-database querying (your 6 APIs!)
Broad initial queries
Include grey literature, preprints
Citation network crawling


Computational Screening

Embedding-based similarity
LLM relevance scoring (abstracts)
Clustering by topic
Reduces to ~500-2000 for deep analysis


Automated Extraction

LLM extracts: methods, datasets, results, claims
Structured data (JSON/database)
Entity recognition (techniques, tools, datasets)
Relationship extraction


Computational Analysis

Topic modeling (BERTopic, LDA)
Trend analysis over time
Citation network analysis
Method co-occurrence patterns
Geographic/institutional patterns


LLM-Powered Synthesis

Automatic summarization by theme
Contradiction detection
Gap identification
Research trajectory mapping
"Ask questions" of corpus


Validation & Writing

Manual spot-checking (sample validation)
Critical papers get full reading
LLM drafts sections
Human edits and interprets



Strengths:

Comprehensive (less selection bias)
Fast (days to weeks vs months)
Can track emerging trends
Reproducible (code-based)
Scalable to rapid fields
"Living review" - easy to update

Limitations:

LLM hallucination risk
Shallower understanding per paper
Quality noise (includes low-quality work)
Requires technical skills
Less established methodology
Validation burden
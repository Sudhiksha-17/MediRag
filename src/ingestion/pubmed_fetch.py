# src/ingestion/pubmed_fetch.py

from Bio import Entrez, Medline
import json
import os
import time

# IMPORTANT: Replace with your actual email
Entrez.email = "sudhikshakandavel@gmail.com"

def fetch_abstracts_for_query(query, max_results=250):
    """
    Fetch PubMed abstracts for a single search query.
    
    How it works:
    1. esearch: searches PubMed and returns a list of paper IDs
    2. efetch: downloads the actual paper details for those IDs
    3. We extract: title, abstract, authors, journal, date
    """
    
    print(f"\n🔍 Searching PubMed for: '{query}'")
    
    # Step A: Search for paper IDs
    try:
        search_handle = Entrez.esearch(
            db="pubmed",           # Search the PubMed database
            term=query,            # Your search query
            retmax=max_results,    # Maximum results to return
            sort="relevance"       # Sort by relevance (most relevant first)
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
    except Exception as e:
        print(f"  ❌ Search failed: {e}")
        return []
    
    paper_ids = search_results["IdList"]
    total_found = search_results["Count"]
    print(f"  Found {total_found} total results, fetching top {len(paper_ids)}")
    
    if not paper_ids:
        print("  ⚠️ No results found for this query")
        return []
    
    # Step B: Fetch paper details in batches of 100
    # (PubMed rate limits: max 100 per request, 3 requests/second)
    all_records = []
    batch_size = 100
    
    for i in range(0, len(paper_ids), batch_size):
        batch_ids = paper_ids[i:i + batch_size]
        print(f"  Fetching batch {i // batch_size + 1}... ({len(batch_ids)} papers)")
        
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=batch_ids,
                rettype="medline",     # MEDLINE format (structured)
                retmode="text"
            )
            records = list(Medline.parse(fetch_handle))
            fetch_handle.close()
            all_records.extend(records)
        except Exception as e:
            print(f"  ❌ Fetch failed for batch: {e}")
        
        # Be polite to PubMed — wait between batches
        time.sleep(0.5)
    
    # Step C: Extract relevant fields from each record
    abstracts = []
    skipped = 0
    
    for record in all_records:
        # Skip papers that don't have abstracts
        if "AB" not in record or not record["AB"].strip():
            skipped += 1
            continue
        
        abstract_entry = {
            "pmid": record.get("PMID", "unknown"),
            "title": record.get("TI", "No title"),
            "abstract": record.get("AB", ""),
            "authors": record.get("AU", []),        # List of author names
            "journal": record.get("JT", "Unknown"), # Full journal title
            "date": record.get("DP", "Unknown"),    # Publication date
            "mesh_terms": record.get("MH", []),      # Medical subject headings
            "keywords": record.get("OT", []),        # Author keywords
        }
        abstracts.append(abstract_entry)
    
    print(f"  ✅ Got {len(abstracts)} abstracts (skipped {skipped} without abstracts)")
    return abstracts


def fetch_all_medical_abstracts():
    """
    Fetch abstracts using multiple targeted queries to get diverse coverage.
    
    WHY MULTIPLE QUERIES?
    A single query like "breast mammography" would miss many relevant papers.
    By using specific sub-topics, you get better coverage of:
    - Different abnormality types (calcifications, masses, distortions)
    - Different techniques (CAD, deep learning, screening)
    - Clinical guidelines and outcomes
    """
    
    queries = [
        # Core mammography topics
        "breast mammography BI-RADS classification",
        "mammogram calcification detection",
        "breast cancer screening mammography findings",
        "mammography mass characterization benign malignant",
        
        # Specific abnormalities
        "mammographic calcification morphology distribution",
        "breast mass margins shape mammography",
        "architectural distortion mammography",
        "asymmetry mammography assessment",
        
        # AI/ML in mammography (relevant to your project)
        "deep learning mammography computer aided detection",
        "convolutional neural network breast cancer detection",
        
        # Clinical guidelines
        "mammography screening guidelines recommendations",
        "breast imaging reporting data system assessment categories",
    ]
    
    all_abstracts = []
    
    for query in queries:
        abstracts = fetch_abstracts_for_query(query, max_results=100)
        all_abstracts.extend(abstracts)
        
        # Wait between queries to avoid rate limiting
        time.sleep(1)
    
    # Deduplicate by PMID (same paper may appear in multiple queries)
    seen_pmids = set()
    unique_abstracts = []
    duplicates = 0
    
    for abstract in all_abstracts:
        pmid = abstract["pmid"]
        if pmid not in seen_pmids:
            seen_pmids.add(pmid)
            unique_abstracts.append(abstract)
        else:
            duplicates += 1
    
    print(f"\n{'='*50}")
    print(f"📊 SUMMARY:")
    print(f"  Total fetched: {len(all_abstracts)}")
    print(f"  Duplicates removed: {duplicates}")
    print(f"  Unique abstracts: {len(unique_abstracts)}")
    print(f"{'='*50}")
    
    return unique_abstracts


def save_abstracts(abstracts, output_path):
    """Save abstracts to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(abstracts)} abstracts to {output_path}")
    
    # Print a sample so you can verify
    if abstracts:
        sample = abstracts[0]
        print(f"\n📄 Sample abstract:")
        print(f"  Title: {sample['title'][:80]}...")
        print(f"  PMID: {sample['pmid']}")
        print(f"  Journal: {sample['journal']}")
        print(f"  Abstract: {sample['abstract'][:150]}...")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🚀 Starting PubMed abstract collection...")
    print("This will take 5-10 minutes.\n")
    
    abstracts = fetch_all_medical_abstracts()
    save_abstracts(abstracts, "data/pubmed_abstracts/abstracts.json")
    
    print("\n✅ Done! Check data/pubmed_abstracts/abstracts.json")
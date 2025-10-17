import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class PubMedArticle:
    """Data class representing a PubMed article"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: datetime
    doi: Optional[str]
    pmc_id: Optional[str]
    publication_types: List[str]
    mesh_terms: List[str]
    keywords: List[str]
    pubmed_url: str
    
    def __str__(self):
        return f"{self.title}\nAuthors: {', '.join(self.authors[:3])}{'...' if len(self.authors) > 3 else ''}\nJournal: {self.journal}\nPublished: {self.publication_date.year}"


class PubMedApi:
    """
    A client for interacting with the PubMed API (NCBI E-utilities).
    
    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    
    Note: Please be respectful of NCBI's usage guidelines:
    - Max 3 requests per second without API key
    - Max 10 requests per second with API key
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize the PubMed API client.
        
        Args:
            api_key: NCBI API key (optional, increases rate limit)
            email: Your email (recommended by NCBI)
        """
        self.api_key = api_key
        self.email = email
        self.last_request_time = 0
        self.min_request_interval = 0.34 if not api_key else 0.1  # Respect rate limits
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _build_params(self, params: Dict) -> Dict:
        """Add common parameters to request"""
        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email
        return params
    
    def search(self,
               query: str,
               max_results: int = 10,
               start: int = 0,
               sort: str = "relevance") -> List[PubMedArticle]:
        """
        Search PubMed for articles.
        
        Args:
            query: Search query (supports PubMed query syntax)
            max_results: Maximum number of results
            start: Starting index for pagination
            sort: Sort order ("relevance", "pub_date", "Author", "JournalName")
            
        Returns:
            List of PubMedArticle objects
        """
        # Step 1: Search to get PMIDs
        pmids = self._search_pmids(query, max_results, start, sort)
        
        if not pmids:
            return []
        
        # Step 2: Fetch details for PMIDs
        return self._fetch_details(pmids)
    
    def _search_pmids(self, query: str, max_results: int, start: int, sort: str) -> List[str]:
        """Search PubMed and return list of PMIDs"""
        self._rate_limit()
        
        params = self._build_params({
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retstart': start,
            'sort': sort,
            'retmode': 'xml'
        })
        
        url = f"{self.BASE_URL}/esearch.fcgi?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
            
            root = ET.fromstring(xml_data)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            return pmids
        
        except Exception as e:
            raise Exception(f"Error searching PubMed: {str(e)}")
    
    def _fetch_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch full details for a list of PMIDs"""
        if not pmids:
            return []
        
        self._rate_limit()
        
        params = self._build_params({
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        })
        
        url = f"{self.BASE_URL}/efetch.fcgi?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
            
            return self._parse_articles(xml_data)
        
        except Exception as e:
            raise Exception(f"Error fetching article details: {str(e)}")
    
    def get_by_pmid(self, pmid: str) -> Optional[PubMedArticle]:
        """
        Get a specific article by PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PubMedArticle object or None
        """
        articles = self._fetch_details([pmid])
        return articles[0] if articles else None
    
    def query(self,
              text: Optional[str] = None,
              author: Optional[str] = None,
              title: Optional[str] = None,
              journal: Optional[str] = None,
              year: Optional[int] = None,
              year_range: Optional[tuple] = None,
              publication_type: Optional[str] = None,
              max_results: int = 10,
              sort: str = "relevance") -> List[PubMedArticle]:
        """
        High-level query interface with natural parameter names.
        
        Args:
            text: General search text
            author: Author name
            title: Keywords in title
            journal: Journal name
            year: Specific publication year
            year_range: Tuple of (start_year, end_year)
            publication_type: Type of publication (e.g., "Review", "Clinical Trial")
            max_results: Maximum number of results
            sort: Sort order ("relevance", "pub_date", "Author", "JournalName")
            
        Returns:
            List of PubMedArticle objects
            
        Examples:
            # Search for papers about CRISPR
            api.query(text="CRISPR gene editing")
            
            # Find papers by author
            api.query(author="Smith J", year=2024)
            
            # Search in specific journal
            api.query(text="cancer immunotherapy", journal="Nature")
            
            # Reviews in a year range
            api.query(text="machine learning", 
                     publication_type="Review",
                     year_range=(2020, 2024))
        """
        query_parts = []
        
        # Build query using PubMed search syntax
        if text:
            query_parts.append(text)
        
        if author:
            query_parts.append(f"{author}[Author]")
        
        if title:
            query_parts.append(f"{title}[Title]")
        
        if journal:
            query_parts.append(f"{journal}[Journal]")
        
        if year:
            query_parts.append(f"{year}[PDAT]")
        elif year_range:
            start_year, end_year = year_range
            query_parts.append(f"{start_year}:{end_year}[PDAT]")
        
        if publication_type:
            query_parts.append(f"{publication_type}[PT]")
        
        if not query_parts:
            raise ValueError("At least one search parameter must be provided")
        
        # Combine with AND
        query_string = " AND ".join(query_parts)
        
        return self.search(query_string, max_results, 0, sort)
    
    def search_by_doi(self, doi: str) -> Optional[PubMedArticle]:
        """Search for an article by DOI"""
        results = self.search(f"{doi}[DOI]", max_results=1)
        return results[0] if results else None
    
    def get_related(self, pmid: str, max_results: int = 10) -> List[PubMedArticle]:
        """Get articles related to a specific PMID"""
        self._rate_limit()
        
        params = self._build_params({
            'dbfrom': 'pubmed',
            'db': 'pubmed',
            'id': pmid,
            'retmax': max_results
        })
        
        url = f"{self.BASE_URL}/elink.fcgi?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
            
            root = ET.fromstring(xml_data)
            related_pmids = [id_elem.text for id_elem in root.findall('.//Link/Id')]
            
            if related_pmids:
                return self._fetch_details(related_pmids[:max_results])
            return []
        
        except Exception as e:
            print(f"Error fetching related articles: {e}")
            return []
    
    def _parse_articles(self, xml_data: str) -> List[PubMedArticle]:
        """Parse XML response containing article details"""
        articles = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article_elem in root.findall('.//PubmedArticle'):
                article = self._parse_article(article_elem)
                if article:
                    articles.append(article)
        
        except Exception as e:
            print(f"Error parsing articles: {e}")
        
        return articles
    
    def _parse_article(self, article_elem) -> Optional[PubMedArticle]:
        """Parse a single article element"""
        try:
            # Get PMID
            pmid = article_elem.findtext('.//PMID', default='')
            
            # Get title
            title = article_elem.findtext('.//ArticleTitle', default='')
            
            # Get abstract
            abstract_parts = article_elem.findall('.//AbstractText')
            abstract = ' '.join([elem.text or '' for elem in abstract_parts])
            
            # Get authors
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                last_name = author_elem.findtext('LastName', default='')
                fore_name = author_elem.findtext('ForeName', default='')
                if last_name or fore_name:
                    authors.append(f"{fore_name} {last_name}".strip())
                elif author_elem.findtext('CollectiveName'):
                    authors.append(author_elem.findtext('CollectiveName'))
            
            # Get journal
            journal = article_elem.findtext('.//Journal/Title', default='')
            if not journal:
                journal = article_elem.findtext('.//MedlineTA', default='')
            
            # Get publication date
            pub_date = self._parse_date(article_elem.find('.//PubDate'))
            
            # Get DOI
            doi = None
            for article_id in article_elem.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            # Get PMC ID
            pmc_id = None
            for article_id in article_elem.findall('.//ArticleId'):
                if article_id.get('IdType') == 'pmc':
                    pmc_id = article_id.text
                    break
            
            # Get publication types
            pub_types = [pt.text for pt in article_elem.findall('.//PublicationType') if pt.text]
            
            # Get MeSH terms
            mesh_terms = [desc.text for desc in article_elem.findall('.//MeshHeading/DescriptorName') if desc.text]
            
            # Get keywords
            keywords = [kw.text for kw in article_elem.findall('.//Keyword') if kw.text]
            
            # Build PubMed URL
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                pmc_id=pmc_id,
                publication_types=pub_types,
                mesh_terms=mesh_terms,
                keywords=keywords,
                pubmed_url=pubmed_url
            )
        
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None
    
    def _parse_date(self, date_elem) -> datetime:
        """Parse publication date from XML element"""
        if date_elem is None:
            return datetime.now()
        
        try:
            year = int(date_elem.findtext('Year', default=str(datetime.now().year)))
            month = date_elem.findtext('Month', default='Jan')
            day = date_elem.findtext('Day', default='1')
            
            # Convert month name to number
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month_num = month_map.get(month[:3], 1)
            day_num = int(day) if day.isdigit() else 1
            
            return datetime(year, month_num, day_num)
        
        except:
            return datetime.now()

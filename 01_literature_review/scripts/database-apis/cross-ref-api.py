import urllib.request
import urllib.parse
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class CrossRefWork:
    """Data class representing a CrossRef work (publication)"""
    doi: str
    title: str
    abstract: Optional[str]
    authors: List[str]
    journal: Optional[str]
    publisher: str
    publication_date: Optional[datetime]
    type: str
    volume: Optional[str]
    issue: Optional[str]
    page: Optional[str]
    issn: Optional[List[str]]
    isbn: Optional[List[str]]
    url: str
    is_referenced_by_count: int
    references_count: int
    
    def __str__(self):
        authors_str = ', '.join(self.authors[:3]) + ('...' if len(self.authors) > 3 else '')
        date_str = self.publication_date.year if self.publication_date else 'N/A'
        return f"{self.title}\nAuthors: {authors_str}\nJournal: {self.journal or 'N/A'}\nYear: {date_str}"


class CrossRefApi:
    """
    A client for interacting with the CrossRef API.
    
    Documentation: https://api.crossref.org/
    
    CrossRef provides metadata for scholarly publications across publishers.
    No API key required, but please be polite with requests.
    """
    
    BASE_URL = "https://api.crossref.org"
    
    def __init__(self, mailto: Optional[str] = None):
        """
        Initialize the CrossRef API client.
        
        Args:
            mailto: Your email (gets you into the "polite pool" with better rate limits)
        """
        self.mailto = mailto
        self.last_request_time = 0
        self.min_request_interval = 0.05  # Be polite with requests
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a request to the CrossRef API"""
        self._rate_limit()
        
        if params is None:
            params = {}
        
        url = f"{self.BASE_URL}/{endpoint}"
        if params:
            url += f"?{urllib.parse.urlencode(params)}"
        
        try:
            request = urllib.request.Request(url)
            request.add_header('Accept', 'application/json')
            
            if self.mailto:
                request.add_header('User-Agent', f'Python CrossRef Client (mailto:{self.mailto})')
            
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            return data
        
        except Exception as e:
            raise Exception(f"Error fetching data from CrossRef: {str(e)}")
    
    def search(self,
               query: str,
               max_results: int = 10,
               offset: int = 0,
               sort: str = "relevance",
               order: str = "desc") -> List[CrossRefWork]:
        """
        Search CrossRef for works.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (max 1000)
            offset: Starting offset for pagination
            sort: Sort field ("relevance", "score", "updated", "deposited", "indexed", "published")
            order: Sort order ("asc" or "desc")
            
        Returns:
            List of CrossRefWork objects
        """
        params = {
            'query': query,
            'rows': min(max_results, 1000),
            'offset': offset,
            'sort': sort,
            'order': order
        }
        
        data = self._make_request('works', params)
        
        if 'message' in data and 'items' in data['message']:
            return [self._parse_work(item) for item in data['message']['items']]
        
        return []
    
    def get_by_doi(self, doi: str) -> Optional[CrossRefWork]:
        """
        Get a specific work by DOI.
        
        Args:
            doi: Digital Object Identifier (with or without https://doi.org/ prefix)
            
        Returns:
            CrossRefWork object or None if not found
        """
        # Clean DOI
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        
        try:
            data = self._make_request(f'works/{clean_doi}')
            
            if 'message' in data:
                return self._parse_work(data['message'])
        
        except Exception as e:
            print(f"Error fetching DOI: {e}")
        
        return None
    
    def query(self,
              text: Optional[str] = None,
              author: Optional[str] = None,
              title: Optional[str] = None,
              doi: Optional[str] = None,
              publisher: Optional[str] = None,
              container_title: Optional[str] = None,
              year: Optional[int] = None,
              year_range: Optional[tuple] = None,
              type: Optional[str] = None,
              max_results: int = 10,
              sort: str = "relevance") -> List[CrossRefWork]:
        """
        High-level query interface with natural parameter names.
        
        Args:
            text: General search text
            author: Author name
            title: Work title
            doi: DOI to search for
            publisher: Publisher name
            container_title: Journal/container title
            year: Specific publication year
            year_range: Tuple of (start_year, end_year)
            type: Work type (e.g., "journal-article", "book-chapter", "proceedings-article")
            max_results: Maximum number of results
            sort: Sort field ("relevance", "score", "updated", "published")
            
        Returns:
            List of CrossRefWork objects
            
        Examples:
            # Search for papers about neural networks
            api.query(text="neural networks")
            
            # Find works by author
            api.query(author="Smith")
            
            # Search in specific journal
            api.query(text="climate change", container_title="Nature")
            
            # Filter by year range and type
            api.query(text="machine learning",
                     type="journal-article",
                     year_range=(2020, 2024))
            
            # Get work by DOI
            api.query(doi="10.1038/nature12345")
        """
        # If DOI is provided, use get_by_doi
        if doi:
            work = self.get_by_doi(doi)
            return [work] if work else []
        
        # Build query using CrossRef field queries
        query_parts = []
        
        if text:
            query_parts.append(text)
        
        params = {}
        
        if author:
            params['query.author'] = author
        
        if title:
            params['query.title'] = title
        
        if publisher:
            params['query.publisher-name'] = publisher
        
        if container_title:
            params['query.container-title'] = container_title
        
        if type:
            params['filter'] = f'type:{type}'
        
        # Handle year filtering
        if year:
            year_filter = f'from-pub-date:{year},until-pub-date:{year}'
            if 'filter' in params:
                params['filter'] += f',{year_filter}'
            else:
                params['filter'] = year_filter
        elif year_range:
            start_year, end_year = year_range
            year_filter = f'from-pub-date:{start_year},until-pub-date:{end_year}'
            if 'filter' in params:
                params['filter'] += f',{year_filter}'
            else:
                params['filter'] = year_filter
        
        # Combine text query
        query_string = ' '.join(query_parts) if query_parts else ''
        
        if not query_string and not params:
            raise ValueError("At least one search parameter must be provided")
        
        # Add other parameters
        params['rows'] = min(max_results, 1000)
        params['sort'] = sort
        params['order'] = 'desc'
        
        if query_string:
            params['query'] = query_string
        
        # Make request
        data = self._make_request('works', params)
        
        if 'message' in data and 'items' in data['message']:
            return [self._parse_work(item) for item in data['message']['items']]
        
        return []
    
    def get_by_issn(self, issn: str, max_results: int = 10) -> List[CrossRefWork]:
        """Get works from a specific journal by ISSN"""
        params = {
            'filter': f'issn:{issn}',
            'rows': max_results
        }
        
        data = self._make_request('works', params)
        
        if 'message' in data and 'items' in data['message']:
            return [self._parse_work(item) for item in data['message']['items']]
        
        return []
    
    def get_by_isbn(self, isbn: str, max_results: int = 10) -> List[CrossRefWork]:
        """Get works with a specific ISBN"""
        params = {
            'filter': f'isbn:{isbn}',
            'rows': max_results
        }
        
        data = self._make_request('works', params)
        
        if 'message' in data and 'items' in data['message']:
            return [self._parse_work(item) for item in data['message']['items']]
        
        return []
    
    def get_journal_info(self, issn: str) -> Optional[Dict]:
        """Get information about a journal by ISSN"""
        try:
            data = self._make_request(f'journals/{issn}')
            return data.get('message')
        except Exception as e:
            print(f"Error fetching journal info: {e}")
            return None
    
    def _parse_work(self, item: Dict) -> CrossRefWork:
        """Parse a work item from the API response"""
        # Get DOI
        doi = item.get('DOI', '')
        
        # Get title
        titles = item.get('title', [])
        title = titles[0] if titles else ''
        
        # Get abstract
        abstract = item.get('abstract')
        
        # Get authors
        authors = []
        author_data = item.get('author', [])
        for author in author_data:
            given = author.get('given', '')
            family = author.get('family', '')
            if given or family:
                authors.append(f"{given} {family}".strip())
        
        # Get journal/container
        container_titles = item.get('container-title', [])
        journal = container_titles[0] if container_titles else None
        
        # Get publisher
        publisher = item.get('publisher', '')
        
        # Get publication date
        pub_date = None
        date_parts = item.get('published', {}).get('date-parts', [[]])
        if date_parts and date_parts[0]:
            parts = date_parts[0]
            year = parts[0] if len(parts) > 0 else None
            month = parts[1] if len(parts) > 1 else 1
            day = parts[2] if len(parts) > 2 else 1
            if year:
                try:
                    pub_date = datetime(year, month, day)
                except:
                    pub_date = datetime(year, 1, 1)
        
        # Get type
        work_type = item.get('type', '')
        
        # Get volume, issue, page
        volume = item.get('volume')
        issue = item.get('issue')
        page = item.get('page')
        
        # Get ISSN and ISBN
        issn = item.get('ISSN', [])
        isbn = item.get('ISBN', [])
        
        # Get URL
        url = item.get('URL', f"https://doi.org/{doi}")
        
        # Get citation counts
        is_referenced_by_count = item.get('is-referenced-by-count', 0)
        references_count = item.get('references-count', 0)
        
        return CrossRefWork(
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publisher=publisher,
            publication_date=pub_date,
            type=work_type,
            volume=volume,
            issue=issue,
            page=page,
            issn=issn,
            isbn=isbn,
            url=url,
            is_referenced_by_count=is_referenced_by_count,
            references_count=references_count
        )


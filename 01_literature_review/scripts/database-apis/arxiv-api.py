
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ArxivPaper:
    """Data class representing an arXiv paper"""
    id: str
    title: str
    summary: str
    authors: List[str]
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: str
    abs_url: str
    
    def __str__(self):
        return f"{self.title}\nAuthors: {', '.join(self.authors)}\nPublished: {self.published.date()}"


class ArxivApi:
    """
    A client for interacting with the arXiv API.
    
    Documentation: https://info.arxiv.org/help/api/index.html
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    NAMESPACE = {'atom': 'http://www.w3.org/2005/Atom',
                 'arxiv': 'http://arxiv.org/schemas/atom'}
    
    def __init__(self):
        """Initialize the arXiv API client"""
        pass
    
    def search(self, 
               query: str,
               max_results: int = 10,
               start: int = 0,
               sort_by: str = "relevance",
               sort_order: str = "descending") -> List[ArxivPaper]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query (e.g., "cat:cs.AI", "au:Smith", "ti:neural networks")
            max_results: Maximum number of results to return
            start: Starting index for results (for pagination)
            sort_by: Sort criterion ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort order ("ascending" or "descending")
            
        Returns:
            List of ArxivPaper objects
        """
        params = {
            'search_query': query,
            'start': start,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
            
            return self._parse_response(xml_data)
        
        except Exception as e:
            raise Exception(f"Error fetching data from arXiv: {str(e)}")
    
    def get_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Get a specific paper by its arXiv ID.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.00001" or "arxiv:2301.00001")
            
        Returns:
            ArxivPaper object or None if not found
        """
        # Clean the ID
        clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        
        results = self.search(f"id:{clean_id}", max_results=1)
        return results[0] if results else None
    
    def search_by_author(self, author: str, max_results: int = 10) -> List[ArxivPaper]:
        """Search for papers by author name"""
        return self.search(f"au:{author}", max_results=max_results)
    
    def search_by_title(self, title: str, max_results: int = 10) -> List[ArxivPaper]:
        """Search for papers by title"""
        return self.search(f"ti:{title}", max_results=max_results)
    
    def search_by_category(self, category: str, max_results: int = 10) -> List[ArxivPaper]:
        """
        Search for papers by category.
        
        Common categories:
        - cs.AI: Artificial Intelligence
        - cs.LG: Machine Learning
        - cs.CL: Computation and Language
        - math.CO: Combinatorics
        - physics.comp-ph: Computational Physics
        """
        return self.search(f"cat:{category}", max_results=max_results)
    
    def query(self, 
              text: Optional[str] = None,
              author: Optional[str] = None,
              title: Optional[str] = None,
              abstract: Optional[str] = None,
              category: Optional[str] = None,
              year: Optional[int] = None,
              max_results: int = 10,
              sort_by: str = "relevance",
              sort_order: str = "descending") -> List[ArxivPaper]:
        """
        High-level query interface with natural parameter names.
        
        Args:
            text: General search text (searches all fields)
            author: Author name to search for
            title: Keywords to search in title
            abstract: Keywords to search in abstract
            category: arXiv category (e.g., "cs.AI", "cs.LG")
            year: Filter by publication year
            max_results: Maximum number of results
            sort_by: Sort criterion ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort order ("ascending" or "descending")
            
        Returns:
            List of ArxivPaper objects
            
        Examples:
            # Search for papers about transformers
            api.query(title="transformer", category="cs.LG")
            
            # Find recent papers by an author
            api.query(author="Vaswani", year=2023)
            
            # General search
            api.query(text="attention mechanisms neural networks")
            
            # Complex query
            api.query(title="bert", author="Devlin", category="cs.CL")
        """
        query_parts = []
        
        # Build query from parameters
        if text:
            query_parts.append(f"all:{text}")
        
        if author:
            query_parts.append(f"au:{author}")
        
        if title:
            query_parts.append(f"ti:{title}")
        
        if abstract:
            query_parts.append(f"abs:{abstract}")
        
        if category:
            query_parts.append(f"cat:{category}")
        
        # Combine query parts
        if not query_parts:
            raise ValueError("At least one search parameter must be provided")
        
        query_string = " AND ".join(query_parts)
        
        # Execute search
        results = self.search(query_string, max_results, 0, sort_by, sort_order)
        
        # Filter by year if specified
        if year and results:
            results = [p for p in results if p.published.year == year]
        
        return results
    
    def _parse_response(self, xml_data: str) -> List[ArxivPaper]:
        """Parse XML response from arXiv API"""
        root = ET.fromstring(xml_data)
        papers = []
        
        for entry in root.findall('atom:entry', self.NAMESPACE):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)
        
        return papers
    
    def _parse_entry(self, entry) -> Optional[ArxivPaper]:
        """Parse a single entry from the XML response"""
        try:
            # Get ID
            id_elem = entry.find('atom:id', self.NAMESPACE)
            paper_id = id_elem.text.split('/abs/')[-1] if id_elem is not None else ""
            
            # Get title
            title_elem = entry.find('atom:title', self.NAMESPACE)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
            
            # Get summary
            summary_elem = entry.find('atom:summary', self.NAMESPACE)
            summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
            
            # Get authors
            authors = []
            for author in entry.findall('atom:author', self.NAMESPACE):
                name_elem = author.find('atom:name', self.NAMESPACE)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Get dates
            published_elem = entry.find('atom:published', self.NAMESPACE)
            published = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00')) if published_elem is not None else datetime.now()
            
            updated_elem = entry.find('atom:updated', self.NAMESPACE)
            updated = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00')) if updated_elem is not None else published
            
            # Get categories
            categories = []
            for category in entry.findall('atom:category', self.NAMESPACE):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Get links
            pdf_url = ""
            abs_url = f"https://arxiv.org/abs/{paper_id}"
            
            for link in entry.findall('atom:link', self.NAMESPACE):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href', '')
            
            return ArxivPaper(
                id=paper_id,
                title=title,
                summary=summary,
                authors=authors,
                published=published,
                updated=updated,
                categories=categories,
                pdf_url=pdf_url,
                abs_url=abs_url
            )
        
        except Exception as e:
            print(f"Error parsing entry: {e}")
            return None
    
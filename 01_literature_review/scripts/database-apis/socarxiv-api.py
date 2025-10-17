import urllib.request
import urllib.parse
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SocArxivPaper:
    """Data class representing a SocArXiv paper"""
    id: str
    title: str
    description: str
    authors: List[str]
    published: datetime
    updated: Optional[datetime]
    tags: List[str]
    subjects: List[str]
    doi: Optional[str]
    pdf_url: Optional[str]
    preprint_url: str
    
    def __str__(self):
        return f"{self.title}\nAuthors: {', '.join(self.authors)}\nPublished: {self.published.date()}"


class SocArxivApi:
    """
    A client for interacting with the SocArXiv API via OSF.
    
    Documentation: https://developer.osf.io/
    """
    
    BASE_URL = "https://api.osf.io/v2"
    SOCARXIV_PROVIDER = "socarxiv"
    
    def __init__(self):
        """Initialize the SocArXiv API client"""
        pass
    
    def search(self,
               query: str = "",
               max_results: int = 10,
               page: int = 1) -> List[SocArxivPaper]:
        """
        Search SocArXiv for preprints.
        
        Args:
            query: Search query string
            max_results: Maximum number of results per page
            page: Page number for pagination
            
        Returns:
            List of SocArxivPaper objects
        """
        params = {
            'filter[provider]': self.SOCARXIV_PROVIDER,
            'page[size]': min(max_results, 100),  # API max is 100
            'page': page
        }
        
        if query:
            params['filter[search]'] = query
        
        url = f"{self.BASE_URL}/preprints/?{urllib.parse.urlencode(params)}"
        
        try:
            request = urllib.request.Request(url)
            request.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            return self._parse_response(data)
        
        except Exception as e:
            raise Exception(f"Error fetching data from SocArXiv: {str(e)}")
    
    def get_by_id(self, preprint_id: str) -> Optional[SocArxivPaper]:
        """
        Get a specific preprint by its ID.
        
        Args:
            preprint_id: SocArXiv preprint ID
            
        Returns:
            SocArxivPaper object or None if not found
        """
        url = f"{self.BASE_URL}/preprints/{preprint_id}/"
        
        try:
            request = urllib.request.Request(url)
            request.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            paper = self._parse_preprint(data['data'])
            return paper
        
        except Exception as e:
            print(f"Error fetching preprint: {e}")
            return None
    
    def search_by_subject(self, subject: str, max_results: int = 10) -> List[SocArxivPaper]:
        """
        Search for preprints by subject area.
        
        Common subjects include: Sociology, Psychology, Political Science, 
        Economics, Education, etc.
        """
        return self.search(subject, max_results=max_results)
    
    def query(self,
              text: Optional[str] = None,
              author: Optional[str] = None,
              title: Optional[str] = None,
              subject: Optional[str] = None,
              year: Optional[int] = None,
              max_results: int = 10,
              page: int = 1) -> List[SocArxivPaper]:
        """
        High-level query interface with natural parameter names.
        
        Args:
            text: General search text
            author: Author name to search for
            title: Keywords to search in title
            subject: Subject area (e.g., "Sociology", "Psychology")
            year: Filter by publication year
            max_results: Maximum number of results
            page: Page number for pagination
            
        Returns:
            List of SocArxivPaper objects
            
        Examples:
            # Search for papers about inequality
            api.query(title="inequality", subject="Sociology")
            
            # Find papers by an author
            api.query(author="Smith")
            
            # General search
            api.query(text="social networks")
            
            # Filter by year
            api.query(text="climate change", year=2024)
        """
        # Build query string
        query_parts = []
        
        if text:
            query_parts.append(text)
        
        if author:
            query_parts.append(author)
        
        if title:
            query_parts.append(title)
        
        if subject:
            query_parts.append(subject)
        
        # Combine query parts
        query_string = " ".join(query_parts) if query_parts else ""
        
        # Execute search
        results = self.search(query_string, max_results, page)
        
        # Filter by year if specified
        if year and results:
            results = [p for p in results if p.published.year == year]
        
        return results
    
    def get_recent(self, max_results: int = 10) -> List[SocArxivPaper]:
        """Get the most recently published preprints"""
        return self.search("", max_results=max_results)
    
    def _parse_response(self, data: Dict) -> List[SocArxivPaper]:
        """Parse JSON response from OSF API"""
        papers = []
        
        if 'data' in data:
            for item in data['data']:
                paper = self._parse_preprint(item)
                if paper:
                    papers.append(paper)
        
        return papers
    
    def _parse_preprint(self, item: Dict) -> Optional[SocArxivPaper]:
        """Parse a single preprint from the API response"""
        try:
            attributes = item.get('attributes', {})
            
            # Get ID
            preprint_id = item.get('id', '')
            
            # Get title
            title = attributes.get('title', '').strip()
            
            # Get description/abstract
            description = attributes.get('description', '').strip()
            
            # Get authors - need to make additional request or use embeds
            # For simplicity, extracting from contributors if available
            authors = []
            contributors = item.get('embeds', {}).get('contributors', {}).get('data', [])
            for contrib in contributors:
                contrib_attrs = contrib.get('attributes', {})
                name = contrib_attrs.get('unregistered_contributor')
                if not name:
                    # Try to get from user data
                    user_data = contrib.get('embeds', {}).get('users', {}).get('data', {})
                    if user_data:
                        user_attrs = user_data.get('attributes', {})
                        full_name = user_attrs.get('full_name', '')
                        if full_name:
                            name = full_name
                
                if name:
                    authors.append(name)
            
            # If no authors found through embeds, try bibliographic_contributors
            if not authors:
                bib_contributors = attributes.get('bibliographic_contributors', [])
                for contrib in bib_contributors:
                    name = contrib.get('name', '')
                    if name:
                        authors.append(name)
            
            # Get dates
            published_str = attributes.get('date_published') or attributes.get('date_created')
            published = datetime.fromisoformat(published_str.replace('Z', '+00:00')) if published_str else datetime.now()
            
            updated_str = attributes.get('date_modified')
            updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00')) if updated_str else None
            
            # Get tags
            tags = attributes.get('tags', [])
            
            # Get subjects
            subjects = []
            subject_data = attributes.get('subjects', [])
            for subj in subject_data:
                if isinstance(subj, dict):
                    subjects.append(subj.get('text', ''))
                elif isinstance(subj, str):
                    subjects.append(subj)
            
            # Get DOI
            doi = attributes.get('doi')
            
            # Get links
            links = item.get('links', {})
            preprint_url = links.get('html', f"https://osf.io/preprints/socarxiv/{preprint_id}/")
            
            # Get PDF URL
            pdf_url = None
            files = item.get('embeds', {}).get('primary_file', {}).get('data', {})
            if files:
                file_links = files.get('links', {})
                pdf_url = file_links.get('download')
            
            return SocArxivPaper(
                id=preprint_id,
                title=title,
                description=description,
                authors=authors,
                published=published,
                updated=updated,
                tags=tags,
                subjects=subjects,
                doi=doi,
                pdf_url=pdf_url,
                preprint_url=preprint_url
            )
        
        except Exception as e:
            print(f"Error parsing preprint: {e}")
            return None

import urllib.request
import urllib.parse
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class OSFPreprint:
    """Data class representing an OSF preprint"""
    id: str
    title: str
    description: str
    authors: List[str]
    created: datetime
    modified: datetime
    published: Optional[datetime]
    provider: str
    subjects: List[str]
    tags: List[str]
    doi: Optional[str]
    license: Optional[str]
    preprint_url: str
    pdf_url: Optional[str]
    
    def __str__(self):
        return f"{self.title}\nAuthors: {', '.join(self.authors) if self.authors else 'N/A'}\nProvider: {self.provider}\nPublished: {self.published.date() if self.published else 'N/A'}"


@dataclass
class OSFProject:
    """Data class representing an OSF project"""
    id: str
    title: str
    description: str
    category: str
    created: datetime
    modified: datetime
    tags: List[str]
    contributors: List[str]
    public: bool
    project_url: str
    
    def __str__(self):
        return f"{self.title}\nCategory: {self.category}\nContributors: {', '.join(self.contributors) if self.contributors else 'N/A'}"


class OSFApi:
    """
    A client for interacting with the OSF (Open Science Framework) API.
    
    Documentation: https://developer.osf.io/
    
    OSF hosts preprints across multiple providers (SocArXiv, PsyArXiv, etc.)
    and research projects with data, code, and materials.
    """
    
    BASE_URL = "https://api.osf.io/v2"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the OSF API client.
        
        Args:
            api_token: Optional API token for authenticated requests (for private content)
        """
        self.api_token = api_token
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Be polite with requests
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a request to the OSF API"""
        self._rate_limit()
        
        if params is None:
            params = {}
        
        url = f"{self.BASE_URL}/{endpoint}"
        if params:
            url += f"?{urllib.parse.urlencode(params)}"
        
        try:
            request = urllib.request.Request(url)
            request.add_header('Accept', 'application/json')
            
            if self.api_token:
                request.add_header('Authorization', f'Bearer {self.api_token}')
            
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            return data
        
        except Exception as e:
            raise Exception(f"Error fetching data from OSF: {str(e)}")
    
    # ========== PREPRINT METHODS ==========
    
    def search_preprints(self,
                        query: str = "",
                        provider: Optional[str] = None,
                        max_results: int = 10,
                        page: int = 1) -> List[OSFPreprint]:
        """
        Search OSF preprints.
        
        Args:
            query: Search query string
            provider: Preprint provider (e.g., "socarxiv", "psyarxiv", "osf", "africarxiv")
            max_results: Maximum number of results per page (max 100)
            page: Page number for pagination
            
        Returns:
            List of OSFPreprint objects
        """
        params = {
            'page[size]': min(max_results, 100),
            'page': page
        }
        
        if query:
            params['filter[search]'] = query
        
        if provider:
            params['filter[provider]'] = provider
        
        data = self._make_request('preprints/', params)
        
        if 'data' in data:
            return [self._parse_preprint(item) for item in data['data']]
        
        return []
    
    def get_preprint_by_id(self, preprint_id: str) -> Optional[OSFPreprint]:
        """
        Get a specific preprint by ID.
        
        Args:
            preprint_id: OSF preprint ID
            
        Returns:
            OSFPreprint object or None if not found
        """
        try:
            data = self._make_request(f'preprints/{preprint_id}/')
            
            if 'data' in data:
                return self._parse_preprint(data['data'])
        
        except Exception as e:
            print(f"Error fetching preprint: {e}")
        
        return None
    
    def query_preprints(self,
                       text: Optional[str] = None,
                       author: Optional[str] = None,
                       title: Optional[str] = None,
                       provider: Optional[str] = None,
                       subject: Optional[str] = None,
                       year: Optional[int] = None,
                       max_results: int = 10) -> List[OSFPreprint]:
        """
        High-level query interface for preprints.
        
        Args:
            text: General search text
            author: Author name
            title: Keywords in title
            provider: Preprint provider (socarxiv, psyarxiv, osf, etc.)
            subject: Subject area
            year: Publication year
            max_results: Maximum number of results
            
        Returns:
            List of OSFPreprint objects
            
        Examples:
            # Search SocArXiv for inequality papers
            api.query_preprints(title="inequality", provider="socarxiv")
            
            # Find PsyArXiv papers by author
            api.query_preprints(author="Smith", provider="psyarxiv")
            
            # General search across all providers
            api.query_preprints(text="climate change")
        """
        query_parts = []
        
        if text:
            query_parts.append(text)
        
        if author:
            query_parts.append(author)
        
        if title:
            query_parts.append(title)
        
        if subject:
            query_parts.append(subject)
        
        query_string = " ".join(query_parts) if query_parts else ""
        
        results = self.search_preprints(query_string, provider, max_results)
        
        # Filter by year if specified
        if year and results:
            results = [p for p in results if p.published and p.published.year == year]
        
        return results
    
    # ========== PROJECT METHODS ==========
    
    def search_projects(self,
                       query: str = "",
                       max_results: int = 10,
                       page: int = 1) -> List[OSFProject]:
        """
        Search OSF projects.
        
        Args:
            query: Search query string
            max_results: Maximum number of results per page (max 100)
            page: Page number for pagination
            
        Returns:
            List of OSFProject objects
        """
        params = {
            'page[size]': min(max_results, 100),
            'page': page
        }
        
        if query:
            params['filter[search]'] = query
        
        data = self._make_request('nodes/', params)
        
        if 'data' in data:
            return [self._parse_project(item) for item in data['data']]
        
        return []
    
    def get_project_by_id(self, project_id: str) -> Optional[OSFProject]:
        """
        Get a specific project by ID.
        
        Args:
            project_id: OSF project ID
            
        Returns:
            OSFProject object or None if not found
        """
        try:
            data = self._make_request(f'nodes/{project_id}/')
            
            if 'data' in data:
                return self._parse_project(data['data'])
        
        except Exception as e:
            print(f"Error fetching project: {e}")
        
        return None
    
    def query_projects(self,
                      text: Optional[str] = None,
                      category: Optional[str] = None,
                      tag: Optional[str] = None,
                      max_results: int = 10) -> List[OSFProject]:
        """
        High-level query interface for projects.
        
        Args:
            text: General search text
            category: Project category (project, data, software, etc.)
            tag: Tag to filter by
            max_results: Maximum number of results
            
        Returns:
            List of OSFProject objects
            
        Examples:
            # Search for machine learning projects
            api.query_projects(text="machine learning")
            
            # Find data projects
            api.query_projects(category="data")
        """
        query_parts = []
        
        if text:
            query_parts.append(text)
        
        if tag:
            query_parts.append(tag)
        
        if category:
            query_parts.append(category)
        
        query_string = " ".join(query_parts) if query_parts else ""
        
        return self.search_projects(query_string, max_results)
    
    def get_project_files(self, project_id: str) -> List[Dict]:
        """
        Get files from a project.
        
        Args:
            project_id: OSF project ID
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            data = self._make_request(f'nodes/{project_id}/files/')
            
            if 'data' in data:
                files = []
                for item in data['data']:
                    attrs = item.get('attributes', {})
                    files.append({
                        'name': attrs.get('name', ''),
                        'kind': attrs.get('kind', ''),
                        'size': attrs.get('size', 0),
                        'path': attrs.get('path', ''),
                        'materialized_path': attrs.get('materialized_path', ''),
                        'download_url': item.get('links', {}).get('download')
                    })
                return files
        
        except Exception as e:
            print(f"Error fetching project files: {e}")
        
        return []
    
    # ========== PROVIDER METHODS ==========
    
    def list_preprint_providers(self) -> List[Dict]:
        """
        List all available preprint providers.
        
        Returns:
            List of provider information dictionaries
        """
        try:
            data = self._make_request('preprint_providers/')
            
            if 'data' in data:
                providers = []
                for item in data['data']:
                    attrs = item.get('attributes', {})
                    providers.append({
                        'id': item.get('id', ''),
                        'name': attrs.get('name', ''),
                        'description': attrs.get('description', ''),
                        'domain': attrs.get('domain', ''),
                        'share_publish_type': attrs.get('share_publish_type', '')
                    })
                return providers
        
        except Exception as e:
            print(f"Error fetching providers: {e}")
        
        return []
    
    # ========== PARSING METHODS ==========
    
    def _parse_preprint(self, item: Dict) -> OSFPreprint:
        """Parse a preprint from the API response"""
        attributes = item.get('attributes', {})
        
        # Get ID
        preprint_id = item.get('id', '')
        
        # Get title
        title = attributes.get('title', '').strip()
        
        # Get description
        description = attributes.get('description', '').strip()
        
        # Get authors (simplified - would need to fetch contributors)
        authors = []
        
        # Get dates
        created_str = attributes.get('date_created')
        created = datetime.fromisoformat(created_str.replace('Z', '+00:00')) if created_str else datetime.now()
        
        modified_str = attributes.get('date_modified')
        modified = datetime.fromisoformat(modified_str.replace('Z', '+00:00')) if modified_str else created
        
        published_str = attributes.get('date_published')
        published = datetime.fromisoformat(published_str.replace('Z', '+00:00')) if published_str else None
        
        # Get provider
        provider_data = item.get('relationships', {}).get('provider', {}).get('data', {})
        provider = provider_data.get('id', 'osf')
        
        # Get subjects
        subjects = []
        subject_data = attributes.get('subjects', [])
        for subj in subject_data:
            if isinstance(subj, dict):
                subjects.append(subj.get('text', ''))
            elif isinstance(subj, str):
                subjects.append(subj)
        
        # Get tags
        tags = attributes.get('tags', [])
        
        # Get DOI
        doi = attributes.get('doi')
        
        # Get license
        license_data = item.get('relationships', {}).get('license', {}).get('data', {})
        license_name = license_data.get('id') if license_data else None
        
        # Get URLs
        links = item.get('links', {})
        preprint_url = links.get('html', f"https://osf.io/{preprint_id}/")
        
        # Get PDF URL (simplified)
        pdf_url = None
        
        return OSFPreprint(
            id=preprint_id,
            title=title,
            description=description,
            authors=authors,
            created=created,
            modified=modified,
            published=published,
            provider=provider,
            subjects=subjects,
            tags=tags,
            doi=doi,
            license=license_name,
            preprint_url=preprint_url,
            pdf_url=pdf_url
        )
    
    def _parse_project(self, item: Dict) -> OSFProject:
        """Parse a project from the API response"""
        attributes = item.get('attributes', {})
        
        # Get ID
        project_id = item.get('id', '')
        
        # Get title
        title = attributes.get('title', '').strip()
        
        # Get description
        description = attributes.get('description', '').strip()
        
        # Get category
        category = attributes.get('category', '')
        
        # Get dates
        created_str = attributes.get('date_created')
        created = datetime.fromisoformat(created_str.replace('Z', '+00:00')) if created_str else datetime.now()
        
        modified_str = attributes.get('date_modified')
        modified = datetime.fromisoformat(modified_str.replace('Z', '+00:00')) if modified_str else created
        
        # Get tags
        tags = attributes.get('tags', [])
        
        # Get contributors (simplified)
        contributors = []
        
        # Get public status
        public = attributes.get('public', False)
        
        # Get URL
        links = item.get('links', {})
        project_url = links.get('html', f"https://osf.io/{project_id}/")
        
        return OSFProject(
            id=project_id,
            title=title,
            description=description,
            category=category,
            created=created,
            modified=modified,
            tags=tags,
            contributors=contributors,
            public=public,
            project_url=project_url
        )


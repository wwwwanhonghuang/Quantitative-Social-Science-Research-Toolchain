import urllib.request
import urllib.parse
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class ZenodoRecord:
    """Data class representing a Zenodo record"""
    id: str
    doi: str
    title: str
    description: str
    creators: List[str]
    publication_date: datetime
    updated: datetime
    resource_type: str
    version: Optional[str]
    keywords: List[str]
    subjects: List[str]
    license: Optional[str]
    access_right: str
    communities: List[str]
    record_url: str
    files: List[Dict]
    
    def __str__(self):
        creators_str = ', '.join(self.creators[:3]) + ('...' if len(self.creators) > 3 else '')
        return f"{self.title}\nCreators: {creators_str}\nType: {self.resource_type}\nPublished: {self.publication_date.date()}"


class ZenodoApi:
    """
    A client for interacting with the Zenodo API.
    
    Documentation: https://developers.zenodo.org/
    
    Zenodo is a research data repository that hosts datasets, software,
    publications, presentations, and other research outputs.
    """
    
    BASE_URL = "https://zenodo.org/api"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the Zenodo API client.
        
        Args:
            api_token: Optional API token for authenticated requests (for private records/uploads)
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
        """Make a request to the Zenodo API"""
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
                params['access_token'] = self.api_token
                url = f"{self.BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
                request = urllib.request.Request(url)
                request.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            return data
        
        except Exception as e:
            raise Exception(f"Error fetching data from Zenodo: {str(e)}")
    
    def search(self,
               query: str = "",
               resource_type: Optional[str] = None,
               max_results: int = 10,
               page: int = 1,
               sort: str = "bestmatch") -> List[ZenodoRecord]:
        """
        Search Zenodo records.
        
        Args:
            query: Search query string (supports Elasticsearch query syntax)
            resource_type: Filter by resource type (dataset, software, publication, etc.)
            max_results: Maximum number of results per page (max 9999)
            page: Page number for pagination
            sort: Sort order ("bestmatch", "mostrecent", "-mostrecent")
            
        Returns:
            List of ZenodoRecord objects
        """
        params = {
            'size': min(max_results, 9999),
            'page': page,
            'sort': sort
        }
        
        if query:
            params['q'] = query
        
        if resource_type:
            params['type'] = resource_type
        
        data = self._make_request('records', params)
        
        if 'hits' in data and 'hits' in data['hits']:
            return [self._parse_record(hit) for hit in data['hits']['hits']]
        
        return []
    
    def get_by_id(self, record_id: str) -> Optional[ZenodoRecord]:
        """
        Get a specific record by ID.
        
        Args:
            record_id: Zenodo record ID
            
        Returns:
            ZenodoRecord object or None if not found
        """
        try:
            data = self._make_request(f'records/{record_id}')
            return self._parse_record(data)
        
        except Exception as e:
            print(f"Error fetching record: {e}")
        
        return None
    
    def get_by_doi(self, doi: str) -> Optional[ZenodoRecord]:
        """
        Get a record by DOI.
        
        Args:
            doi: DOI (with or without https://doi.org/ prefix)
            
        Returns:
            ZenodoRecord object or None if not found
        """
        # Clean DOI
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        
        results = self.search(f'doi:"{clean_doi}"', max_results=1)
        return results[0] if results else None
    
    def query(self,
              text: Optional[str] = None,
              creator: Optional[str] = None,
              title: Optional[str] = None,
              description: Optional[str] = None,
              keywords: Optional[str] = None,
              resource_type: Optional[str] = None,
              community: Optional[str] = None,
              year: Optional[int] = None,
              year_range: Optional[tuple] = None,
              access_right: Optional[str] = None,
              max_results: int = 10,
              sort: str = "bestmatch") -> List[ZenodoRecord]:
        """
        High-level query interface with natural parameter names.
        
        Args:
            text: General search text (searches all fields)
            creator: Creator/author name
            title: Keywords in title
            description: Keywords in description
            keywords: Keywords to search for
            resource_type: Type of resource (dataset, software, publication, image, video, etc.)
            community: Community name
            year: Specific publication year
            year_range: Tuple of (start_year, end_year)
            access_right: Access level (open, embargoed, restricted, closed)
            max_results: Maximum number of results
            sort: Sort order ("bestmatch", "mostrecent", "-mostrecent")
            
        Returns:
            List of ZenodoRecord objects
            
        Examples:
            # Search for datasets about climate
            api.query(text="climate", resource_type="dataset")
            
            # Find software by creator
            api.query(creator="Smith", resource_type="software")
            
            # Search in specific community
            api.query(text="machine learning", community="zenodo")
            
            # Open access publications from 2023
            api.query(title="neural networks",
                     resource_type="publication",
                     year=2023,
                     access_right="open")
        """
        query_parts = []
        
        # Build query using Zenodo search syntax
        if text:
            query_parts.append(text)
        
        if creator:
            query_parts.append(f'creators.name:"{creator}"')
        
        if title:
            query_parts.append(f'title:"{title}"')
        
        if description:
            query_parts.append(f'description:"{description}"')
        
        if keywords:
            query_parts.append(f'keywords:"{keywords}"')
        
        if community:
            query_parts.append(f'communities:"{community}"')
        
        if year:
            query_parts.append(f'publication_date:[{year}-01-01 TO {year}-12-31]')
        elif year_range:
            start_year, end_year = year_range
            query_parts.append(f'publication_date:[{start_year}-01-01 TO {end_year}-12-31]')
        
        if access_right:
            query_parts.append(f'access_right:{access_right}')
        
        # Combine query parts
        query_string = " AND ".join(query_parts) if query_parts else ""
        
        return self.search(query_string, resource_type, max_results, 1, sort)
    
    def search_datasets(self, query: str = "", max_results: int = 10) -> List[ZenodoRecord]:
        """Search for datasets"""
        return self.search(query, resource_type="dataset", max_results=max_results)
    
    def search_software(self, query: str = "", max_results: int = 10) -> List[ZenodoRecord]:
        """Search for software"""
        return self.search(query, resource_type="software", max_results=max_results)
    
    def search_publications(self, query: str = "", max_results: int = 10) -> List[ZenodoRecord]:
        """Search for publications"""
        return self.search(query, resource_type="publication", max_results=max_results)
    
    def search_by_community(self, community: str, max_results: int = 10) -> List[ZenodoRecord]:
        """
        Search records in a specific community.
        
        Args:
            community: Community identifier
            max_results: Maximum number of results
        """
        return self.search(f'communities:"{community}"', max_results=max_results)
    
    def get_communities(self) -> List[Dict]:
        """
        Get list of Zenodo communities.
        
        Returns:
            List of community information dictionaries
        """
        try:
            data = self._make_request('communities')
            
            if 'hits' in data and 'hits' in data['hits']:
                communities = []
                for hit in data['hits']['hits']:
                    communities.append({
                        'id': hit.get('id', ''),
                        'title': hit.get('title', ''),
                        'description': hit.get('description', ''),
                        'page': hit.get('page', ''),
                        'curation_policy': hit.get('curation_policy', '')
                    })
                return communities
        
        except Exception as e:
            print(f"Error fetching communities: {e}")
        
        return []
    
    def get_record_versions(self, record_id: str) -> List[ZenodoRecord]:
        """
        Get all versions of a record.
        
        Args:
            record_id: Zenodo record ID
            
        Returns:
            List of ZenodoRecord objects representing different versions
        """
        try:
            # First get the record to find the concept DOI
            record = self.get_by_id(record_id)
            if not record:
                return []
            
            # Search for all versions using the concept DOI
            # This is a simplified approach
            return [record]
        
        except Exception as e:
            print(f"Error fetching versions: {e}")
        
        return []
    
    def _parse_record(self, item: Dict) -> ZenodoRecord:
        """Parse a record from the API response"""
        metadata = item.get('metadata', {})
        
        # Get ID
        record_id = str(item.get('id', ''))
        
        # Get DOI
        doi = metadata.get('doi', '')
        
        # Get title
        title = metadata.get('title', '').strip()
        
        # Get description
        description = metadata.get('description', '').strip()
        
        # Get creators
        creators = []
        creator_data = metadata.get('creators', [])
        for creator in creator_data:
            name = creator.get('name', '')
            if name:
                creators.append(name)
        
        # Get publication date
        pub_date_str = metadata.get('publication_date')
        pub_date = datetime.fromisoformat(pub_date_str) if pub_date_str else datetime.now()
        
        # Get updated date
        updated_str = item.get('updated')
        updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00')) if updated_str else pub_date
        
        # Get resource type
        resource_type_data = metadata.get('resource_type', {})
        resource_type = resource_type_data.get('type', 'unknown')
        if 'subtype' in resource_type_data:
            resource_type = f"{resource_type}:{resource_type_data['subtype']}"
        
        # Get version
        version = metadata.get('version')
        
        # Get keywords
        keywords = metadata.get('keywords', [])
        
        # Get subjects
        subjects = []
        subject_data = metadata.get('subjects', [])
        for subj in subject_data:
            if isinstance(subj, dict):
                subjects.append(subj.get('term', ''))
            elif isinstance(subj, str):
                subjects.append(subj)
        
        # Get license
        license_data = metadata.get('license', {})
        license_id = license_data.get('id') if isinstance(license_data, dict) else None
        
        # Get access right
        access_right = metadata.get('access_right', 'unknown')
        
        # Get communities
        communities = []
        community_data = metadata.get('communities', [])
        for comm in community_data:
            if isinstance(comm, dict):
                communities.append(comm.get('id', ''))
            elif isinstance(comm, str):
                communities.append(comm)
        
        # Get record URL
        links = item.get('links', {})
        record_url = links.get('html', f"https://zenodo.org/record/{record_id}")
        
        # Get files
        files = []
        file_data = item.get('files', [])
        for file_item in file_data:
            files.append({
                'key': file_item.get('key', ''),
                'size': file_item.get('size', 0),
                'checksum': file_item.get('checksum', ''),
                'type': file_item.get('type', ''),
                'download_url': file_item.get('links', {}).get('self', '')
            })
        
        return ZenodoRecord(
            id=record_id,
            doi=doi,
            title=title,
            description=description,
            creators=creators,
            publication_date=pub_date,
            updated=updated,
            resource_type=resource_type,
            version=version,
            keywords=keywords,
            subjects=subjects,
            license=license_id,
            access_right=access_right,
            communities=communities,
            record_url=record_url,
            files=files
        )


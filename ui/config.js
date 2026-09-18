// The 'web' core service is published on a different host port than this
// static site (see docker-compose.yml) - derived from the current page's
// hostname rather than hardcoding "localhost", so this also works when the
// demo is opened from another machine on the same network via a LAN IP.
const API_BASE = `${window.location.protocol}//${window.location.hostname}:8001`;

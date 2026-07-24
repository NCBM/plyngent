from .fetch import fetch as fetch
from .grants import clear_private_grants as clear_private_grants
from .grants import grant_private_host as grant_private_host
from .grants import set_fetch_policy_confirm_hook as set_fetch_policy_confirm_hook
from .policy import clear_ssrf_assume_public_cidrs as clear_ssrf_assume_public_cidrs
from .policy import set_ssrf_assume_public_cidrs as set_ssrf_assume_public_cidrs

NET_TOOLS = [
    fetch,
]

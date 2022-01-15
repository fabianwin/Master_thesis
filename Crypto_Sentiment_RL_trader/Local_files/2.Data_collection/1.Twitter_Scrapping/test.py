from pytrends.request import TrendReq as UTrendReq
GET_METHOD='get'

import requests

headers = {
...
}


class TrendReq(UTrendReq):
    def _get_data(self, url, method=GET_METHOD, trim_chars=0, **kwargs):
        return super()._get_data(url, method=GET_METHOD, trim_chars=trim_chars, headers=headers, **kwargs)

from pytrends.request import TrendReq as UTrendReq
import requests

GET_METHOD='get'


headers = {
    'authority': 'trends.google.com',
    'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
    'accept': 'application/json, text/plain, */*',
    'sec-ch-ua-mobile': '?0',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
    'x-client-data': 'CIq2yQEIpLbJAQjEtskBCKmdygEIlu/KAQjr8ssBCJ75ywEI1/zLAQjnhMwBCLaFzAEIrI7MAQjSj8wBCNqQzAEI2pPMARiMnssB',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://trends.google.com/trends/explore?date=2017-01-01%202017-12-31&q=Cardano',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,de;q=0.7',
    'cookie': '__utma=10102256.1276915032.1642254453.1642254453.1642254453.1; __utmc=10102256; __utmz=10102256.1642254453.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utmt=1; __utmb=10102256.28.7.1642254700832; CONSENT=YES+CH.de+20160925-15-0; ANID=AHWqTUl2c94oL-Po2XqTTWVhIOSNm0NmVRCGAklHch-163kU34O9TPP6f2B6wl0O; 1P_JAR=2022-01-15-13; SID=GAjwPcJTmIYp0Hp7A9b7EqsGk82CqhVwj7Eu1NGR0ouQOArp-qjFvCclH6ybO4z4IVToBQ.; __Secure-1PSID=GAjwPcJTmIYp0Hp7A9b7EqsGk82CqhVwj7Eu1NGR0ouQOArpx8guv59G9_IO6ar3R2J_Jg.; __Secure-3PSID=GAjwPcJTmIYp0Hp7A9b7EqsGk82CqhVwj7Eu1NGR0ouQOArpa4WOZJ9T1aN0NPFV1VRgeg.; HSID=ALFwtMXdEIJTDERq4; SSID=Av1NbbJyTm6kUqNXp; APISID=Xdns7dMpEvc_9veC/AX-fk56lNmXFVn1UY; SAPISID=ys_DVVn_XvrRNFCe/AERKyhW3eBSOsqf3h; __Secure-1PAPISID=ys_DVVn_XvrRNFCe/AERKyhW3eBSOsqf3h; __Secure-3PAPISID=ys_DVVn_XvrRNFCe/AERKyhW3eBSOsqf3h; NID=511=itnGWrhKnMOYlgOLhqX2iBO5FFeMVKEdINjesfgbedzu-wwPLTKBB3PUfWHEAyq2wRXFzMF6tKAcMEunJQGAIk_saeh48QHuHsORUQNiFDVoh_xMUE4ZIuIM2itF-mZwz4dRdUVGu9DpKx1pH-MrSi83HUS5Z8JKF1u_fuUJQZCoY6EBZz2EJFj8WWO5b-X4IG_3Rcm5lViSw0ZGgdgDlX-ardjp5OzQVtTeX-4; SIDCC=AJi4QfF8cq8gfVotEX3mHgqVWupbCul-FmcDPRc9ePWgWINv3CawF5QKRbfIJxKletygiLVbhw; __Secure-3PSIDCC=AJi4QfFyDKbPn3VmH4Utdjkda3ImyZshVEBFaofJ0WsNIUumh7qAep-S8p3xeSwtPiUjx2NDZQ',
}

params = (
    ('hl', 'en-US'),
    ('tz', '-60'),
    ('req', '{"restriction":{"geo":{},"time":"2017-01-01 2017-12-31","originalTimeRangeForExploreUrl":"2017-01-01 2017-12-31","complexKeywordsRestriction":{"keyword":[{"type":"BROAD","value":"Cardano"}]}},"keywordType":"ENTITY","metric":["TOP","RISING"],"trendinessSettings":{"compareTime":"2016-01-02 2016-12-31"},"requestOptions":{"property":"","backend":"IZG","category":0},"language":"en","userCountryCode":"CH"}'),
    ('token', 'APP6_UEAAAAAYeQqxUkgdmMIza-DXTBZUSZxJPWmx2-g'),
)

response = requests.get('https://trends.google.com/trends/api/widgetdata/relatedsearches', headers=headers, params=params)


class TrendReq(UTrendReq):
    def _get_data(self, url, method=GET_METHOD, trim_chars=0, **kwargs):
        return super()._get_data(url, method=GET_METHOD, trim_chars=trim_chars, headers=headers, **kwargs)

#NB. Original query string below. It seems impossible to parse and
#reproduce query strings 100% accurately so the one below is given
#in case the reproduced version is not "correct".
# response = requests.get('https://trends.google.com/trends/api/widgetdata/relatedsearches?hl=en-US&tz=-60&req=%7B%22restriction%22:%7B%22geo%22:%7B%7D,%22time%22:%222017-01-01+2017-12-31%22,%22originalTimeRangeForExploreUrl%22:%222017-01-01+2017-12-31%22,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22Cardano%22%7D%5D%7D%7D,%22keywordType%22:%22ENTITY%22,%22metric%22:%5B%22TOP%22,%22RISING%22%5D,%22trendinessSettings%22:%7B%22compareTime%22:%222016-01-02+2016-12-31%22%7D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22IZG%22,%22category%22:0%7D,%22language%22:%22en%22,%22userCountryCode%22:%22CH%22%7D&token=APP6_UEAAAAAYeQqxUkgdmMIza-DXTBZUSZxJPWmx2-g', headers=headers)

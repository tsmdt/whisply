NOSCRIBE_HTML_TEMPLATE = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html>
<head>
<meta charset="UTF-8" />
<style type="text/css">
    p, li {white-space: pre-wrap;}
    p {font-size: 0.9em;}
    .MsoNormal {font-family: "Arial"; font-weight: 400; font-style: normal; font-size: 0.9em;}
    @page WordSection1 {mso-line-numbers-restart: continuous; mso-line-numbers-count-by: 1; mso-line-numbers-start: 1;}
    div.WordSection1 {page: WordSection1;}
</style>
<meta name="audio_source" content="{audio_filepath}">
</head>
<body style="font-family: 'Arial'; font-weight: 400; font-style: normal">
<div class="WordSection1">
    <p style="font-weight:600;">{transcription}</p>
    <p style="color: #909090; font-size: 0.8em" >Transcribed with <strong>whisply</strong> for <strong>noScribe's Editor</strong>.<br />Audio source: {audio_filepath}</span></p>
    {body_content}
</div>
</body>
</html>
"""
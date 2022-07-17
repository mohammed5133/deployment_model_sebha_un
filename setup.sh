mkdir -p ~/.streamlit/

echo "\
[grneral]\n
email = \"maha.bargog@fit.sebhau.edu.ly\"\n
">~/.streamlit/credentials.toml
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
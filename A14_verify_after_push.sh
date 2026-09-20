#!/bin/bash
# Re-verify the companion repository after the push, per the README's own step 6:
# "A push that reports success and a repository that matches what you meant to
# send are different claims."
#
# Run from the folder that contains this script and the uploaded tree.
B="https://raw.githubusercontent.com/machyman/hyman2026velocity/main"
ok=0; bad=0; miss=0
while read -r f; do
  [ "$f" = "./A14_verify_after_push.sh" ] && continue
  p="${f#./}"
  code=$(curl -sS -o /tmp/v -w "%{http_code}" "$B/$p" 2>/dev/null)
  if [ "$code" != "200" ]; then printf "MISSING  %s\n" "$p"; miss=$((miss+1)); continue; fi
  a=$(sha256sum "$f"   | cut -c1-16)
  b=$(sha256sum /tmp/v | cut -c1-16)
  if [ "$a" = "$b" ]; then ok=$((ok+1))
  else printf "MISMATCH %s   local %s  live %s\n" "$p" "$a" "$b"; bad=$((bad+1)); fi
done < <(find . -type f | sort)
echo
echo "match $ok   mismatch $bad   missing $miss"
[ $bad -eq 0 ] && [ $miss -eq 0 ] && echo "PUSH VERIFIED" || echo "PUSH NOT VERIFIED"

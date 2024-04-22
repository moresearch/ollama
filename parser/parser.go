package parser

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
)

type Command struct {
	Name string
	Args string
	bytes.Buffer
}

func Parse(r io.Reader) ([]Command, error) {
	var cmds []Command
	var cmd Command
	var b bytes.Buffer

	var quotes int

	s := stateName
	br := bufio.NewReader(r)
	for {
		r, _, err := br.ReadRune()
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}

		if _, err := cmd.WriteRune(r); err != nil {
			return nil, err
		}

		// trim leading whitespace
		if (space(r) || newline(r)) && b.Len() == 0 {
			continue
		}

		switch s {
		case stateName, stateParameter:
			if alpha(r) || number(r) {
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			} else if space(r) {
				cmd.Name = strings.ToLower(b.String())
				b.Reset()

				if cmd.Name == "from" {
					cmd.Name = "model"
				}

				switch cmd.Name {
				case "parameter":
					s = stateParameter
				case "message":
					s = stateMessage
				default:
					s = stateArgs
				}
			} else if newline(r) {
				return nil, fmt.Errorf("missing value for [%s]", b.String())
			}
		case stateArgs:
			if r == '"' && b.Len() == 0 {
				quotes++
				s = stateMultiline
			} else if newline(r) {
				cmd.Args = b.String()
				b.Reset()

				cmds = append(cmds, cmd)
				cmd = Command{}
				s = stateName
			} else {
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			}
		case stateMultiline:
			if r == '"' && b.Len() == 0 {
				quotes++
				continue
			} else if r == '"' {
				if quotes--; quotes == 0 {
					cmd.Args = b.String()
					b.Reset()

					cmds = append(cmds, cmd)
					cmd = Command{}
					s = stateName
				}

				continue
			} else {
				if _, err := b.WriteRune(r); err != nil {
					return nil, err
				}
			}
		case stateMessage:
			if space(r) && !isValidRole(b.String()) {
				return nil, errors.New("role must be one of \"system\", \"user\", or \"assistant\"")
			} else if space(r) {
				if _, err := b.WriteRune(':'); err != nil {
					return nil, err
				}
				s = stateArgs
			}

			if _, err := b.WriteRune(r); err != nil {
				return nil, err
			}
		}
	}

	for _, cmd := range cmds {
		if cmd.Name == "model" {
			return cmds, nil
		}
	}

	return nil, errors.New("no FROM line")
}

const (
	stateName = iota
	stateArgs
	stateMultiline
	stateParameter
	stateMessage
)

func alpha(r rune) bool {
	return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z'
}

func number(r rune) bool {
	return r >= '0' && r <= '9'
}

func space(r rune) bool {
	return r == ' ' || r == '\t'
}

func newline(r rune) bool {
	return r == '\r' || r == '\n'
}

func isValidRole(role string) bool {
	return role == "system" || role == "user" || role == "assistant"
}

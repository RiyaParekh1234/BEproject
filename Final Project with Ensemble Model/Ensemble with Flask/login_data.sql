CREATE DATABASE IF NOT EXISTS `pythonlogin`;
USE `pythonlogin`;

CREATE TABLE accounts (
	id int(11) NOT NULL AUTO_INCREMENT,
  	username varchar(50) NOT NULL,
  	pass varchar(255) NOT NULL,
  	email varchar(100) NOT NULL,
    PRIMARY KEY (id)
);

insert INTO accounts(id, username, pass, email) VALUES (1, 'test', 'test', 'test@test.com');

drop table accounts;
show databases;
DROP database pythonlogin;